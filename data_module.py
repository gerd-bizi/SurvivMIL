# data_module.py
from __future__ import annotations
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
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
        self._combined_csv = None

    def _prepare_csv(self) -> str | None:
        """Resolve the CSV path.

        ``self.csv_path`` can either point directly to a CSV file or to a
        directory containing ``fold{n}_train.csv`` and ``fold{n}_test.csv``
        sub-folders.  When a directory is supplied the train CSV is further
        split into an 80/20 train/val split.
        """

        if self.csv_path is None:
            return None

        if os.path.isdir(self.csv_path):
            fold = getattr(self.args, "fold", 1)
            fold_dir = Path(self.csv_path) / f"fold{fold}"
            train_csv = fold_dir / f"fold{fold}_train.csv"
            test_csv = fold_dir / f"fold{fold}_test.csv"

            train_df = pd.read_csv(train_csv)
            test_df = pd.read_csv(test_csv)

            # Create split column and a validation subset
            val_df = train_df.sample(frac=0.2, random_state=42)
            train_df = train_df.drop(val_df.index)
            train_df["split"] = "train"
            val_df["split"] = "val"
            test_df["split"] = "test"

            combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
            fd, tmp_path = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            combined.to_csv(tmp_path, index=False)
            self._combined_csv = tmp_path
            return tmp_path

        return self.csv_path

    def setup(self, stage=None):
        if self.task == "PredictOutcome":
            csv = self._prepare_csv()

            self.train_dset = histodata(
                concat=self.concat,
                h5_path=self.h5_path,
                csv_path=csv,
                state="train",
                shuffle=self.shuffle,
            )
            self.val_dset = histodata(
                concat=self.concat,
                h5_path=self.h5_path,
                csv_path=csv,
                state="val",
                shuffle=False,
            )
            self.test_dset = histodata(
                concat=self.concat,
                h5_path=self.h5_path,
                csv_path=csv,
                state="test",
                shuffle=False,
            )
            print(
                f"Finished loading datasets: "
                f"{len(self.train_dset)} train / "
                f"{len(self.val_dset)} val / "
                f"{len(self.test_dset)} test bags"
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
