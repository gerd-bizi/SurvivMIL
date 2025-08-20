# data_module.py
from __future__ import annotations
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch.utils.data import DataLoader

from dataset import histodata


def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    return torch.utils.data.dataloader.default_collate(batch)


class histo_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path=None,          # <- interpret as features_dir
        csv_path=None,
        state=None,
        shuffle=True,
        batch_size=1,
        task="PredictOutcome",
        augment_type=None,
        sub_aug_type=None,
        args=None,
        concat=None,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.csv_path = csv_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.task = task
        self.augment_type = augment_type
        self.sub_aug_type = sub_aug_type
        self.args = args
        self.concat = concat

    def setup(self, stage=None):
        if self.task == "PredictOutcome":
            self.train_dset = histodata(
                concat=self.concat,
                h5_path=self.h5_path,
                csv_path=self.csv_path,
                state="train",
                shuffle=self.shuffle,
            )
            self.val_dset = histodata(
                concat=self.concat,
                h5_path=self.h5_path,
                csv_path=self.csv_path,
                state="val",
                shuffle=False,
            )
            self.test_dset = histodata(
                concat=self.concat,
                h5_path=self.h5_path,
                csv_path=self.csv_path,
                state="test",
                shuffle=False,
            )

    def calculate_weights(self):
        # NOTE: This assumes d[1][0] is a single-class label tensor.
        print("Calculating weights for weighted random sampler")
        dloader = DataLoader(self.train_dset, batch_size=1, shuffle=False)
        labels = []
        for d in dloader:
            labels.append(d[1][0].item())
        labels = np.asarray(labels, dtype=int)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        weights = class_weights[labels]
        print("Weights calculated")
        return torch.from_numpy(weights.astype(np.float32))

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.WeightedRandomSampler(
               weights=self.calculate_weights(),
               num_samples=len(self.train_dset),
               replacement=True,
            ),
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=31,
        )
