# dataset.py
from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from bags_io import load_bag, resolve_bag_path


class Dataset_All_Bags(Dataset):
    """Minimal helper kept for compatibility."""
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        # Prefer a standard 'slide_id' column; fall back to the first column if named '0'
        if "slide_id" in self.df.columns:
            return self.df.loc[idx, "slide_id"]
        return self.df.iloc[idx, 0]


class histodata(Dataset):
    """
    Feature-bag dataset that loads (N, D) bags (typically D=1024 for UNI),
    with optional coords (N, 2). No direct .h5 / .pt reading; everything goes
    through survivmil.bags_io which prefers .npy.
    """

    def __init__(
        self,
        h5_path: str = None,   # interpret as features_dir
        csv_path: str = None,
        state: str = None,     # 'train'|'val'|'test'
        shuffle: bool = False,
        one_vs_target: str = 'high',
        concat: bool = False,
    ):
        self.features_dir = h5_path
        self.csv_path = csv_path
        self.state = state
        self.shuffle_instances = shuffle
        self.one_vs_target = one_vs_target
        self.concat = concat

        self.df = pd.read_csv(self.csv_path)

        # Map the CSV schema used in this project to the column names expected
        # by the original code. ``recurrence_bin`` carries both the
        # classification label and the survival event indicator while
        # ``duration_months`` stores the follow-up time.  A spurious
        # ``wsi_placeholder`` column can appear and is ignored.
        # This normalization allows downstream code to consume clinical CSVs
        # without needing project-specific preprocessing hooks.
        if "wsi_placeholder" in self.df.columns:
            self.df = self.df.drop(columns=["wsi_placeholder"])
        if "recurrence_bin" in self.df.columns:
            if "label" not in self.df.columns:
                self.df["label"] = self.df["recurrence_bin"].astype(int)
            if "outcome" not in self.df.columns:
                self.df["outcome"] = self.df["recurrence_bin"].astype(int)
        if "duration_months" in self.df.columns and "survival_days" not in self.df.columns:
            # Keep the units (months) but expose under the expected name.
            self.df["survival_days"] = self.df["duration_months"].astype(float)

        # ---- Split handling -------------------------------------------------
        # Accept either 'Splits' (your legacy) or 'split' (modern).
        split_col = "Splits" if "Splits" in self.df.columns else ("split" if "split" in self.df.columns else None)
        if split_col:
            split_mask = self.df[split_col].astype(str).str.lower().eq(self.state)
            self.df = self.df.loc[split_mask].reset_index(drop=True)
        # else: use the whole CSV as this split (caller is responsible)

        # ---- ID / bag stem handling ----------------------------------------
        # Prefer an explicit 'slide_id' or 'patient_id'. If missing, fall back to a path column named '0'.
        if "slide_id" in self.df.columns:
            id_series = self.df["slide_id"].astype(str)
        elif "patient_id" in self.df.columns:
            # Patient IDs in the fold CSVs are of the form ``x_y`` while the
            # feature directories are prefixed with ``TMA_``.  Normalise here so
            # downstream code always works with the ``TMA_x_y`` form.
            id_series = self.df["patient_id"].astype(str).apply(
                lambda s: s if s.startswith("TMA_") else f"TMA_{s}"
            )
        elif "0" in self.df.columns:
            # Column '0' historically stored a path to a file. We use its basename as the bag stem.
            id_series = self.df["0"].astype(str).apply(lambda p: Path(p).stem)
        else:
            raise ValueError("CSV must contain one of: 'slide_id', 'patient_id', or a path column named '0'.")

        # Normalised bag stems and pruning of missing feature directories/files
        self.df["bag_stem"] = id_series
        stems = self.df["bag_stem"].unique().tolist()
        existing, missing = [], []
        for stem in stems:
            try:
                resolve_bag_path(self.features_dir, stem)
            except FileNotFoundError:
                missing.append(stem)
            else:
                existing.append(stem)
        if missing:
            # Drop rows for patients with no corresponding features
            self.df = self.df[~self.df["bag_stem"].isin(missing)].reset_index(drop=True)
        self.bag_stems = existing

        if self.shuffle_instances:
            random.shuffle(self.bag_stems)

        # ---- Label / target columns (legacy names preserved) ----------------
        # For classification:
        #   'label'  -> class label (int)
        #   'outcome' -> event; in original code: censorship = 1 - outcome
        # For survival:
        #   'survival_days' -> time_to_event
        # Your CSV might differ; adapt here if needed.
        self.has_label = "label" in self.df.columns
        self.has_outcome = "outcome" in self.df.columns
        self.has_time = "survival_days" in self.df.columns

        # Determine which columns will be treated as EHR features.  Exclude ID
        # and target columns, keeping only numeric data.
        reserved = {
            "slide_id",
            "patient_id",
            "bag_stem",
            "Splits",
            "split",
            "label",
            "outcome",
            "survival_days",
            "recurrence_bin",
            "duration_months",
        }
        self.ehr_cols = [
            c
            for c in self.df.columns
            if c not in reserved and np.issubdtype(self.df[c].dtype, np.number)
        ]

    def __len__(self):
        return len(self.bag_stems)

    def _row_by_stem(self, stem: str) -> pd.Series:
        # Try to match stem against slide_id/patient_id first, then fallback to path column '0'
        if "slide_id" in self.df.columns:
            row = self.df[self.df["slide_id"].astype(str) == stem]
        elif "patient_id" in self.df.columns:
            # ``stem`` may have the ``TMA_`` prefix; remove it when matching to
            # the raw ``patient_id`` column.
            clean = stem[4:] if stem.startswith("TMA_") else stem
            row = self.df[self.df["patient_id"].astype(str) == clean]
        elif "0" in self.df.columns:
            row = self.df[self.df["0"].astype(str).apply(lambda p: Path(p).stem) == stem]
        else:
            raise RuntimeError("Unexpected: no ID columns present at query time.")
        if row.empty:
            raise KeyError(f"No CSV row for bag stem '{stem}'")
        return row.iloc[0]

    def _ehr_from_row(self, row: pd.Series) -> torch.Tensor:
        """Extract an EHR feature vector for a patient.

        All numeric columns not reserved for IDs or targets are included.  Any
        missing values are filled with zeros to keep tensor shapes consistent.
        """
        if not getattr(self, "ehr_cols", None):
            return torch.zeros(0, dtype=torch.float32)
        vals = row[self.ehr_cols].astype(float).fillna(0).to_numpy()
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx: int):
        bag_stem = self.bag_stems[idx]
        # Resolve to a .npy first, fallback to .h5/.hdf5 (no direct h5py calls here)
        bag_path = resolve_bag_path(self.features_dir, bag_stem)
        arr, coords = load_bag(bag_path)  # arr is (N,D) or (N,H,W[,C]); we treat it as features

        # Convert to torch tensors (double to match your existing downstream expectations)
        feats = torch.from_numpy(np.asarray(arr)).double()
        coords_t = None if coords is None else torch.from_numpy(np.asarray(coords)).double()

        # Labels / outcomes / time
        row = self._row_by_stem(bag_stem)

        if self.has_label:
            label = np.asarray([int(row["label"])], dtype=np.int64)
        else:
            # If no 'label' column, default 0 so WeightedRandomSampler doesn't crash; adjust for survival workflows
            label = np.asarray([0], dtype=np.int64)

        if self.has_outcome:
            censorship = 1 - int(row["outcome"])
        else:
            censorship = 0

        if self.has_time:
            time_to_event = float(row["survival_days"])
        else:
            time_to_event = 0.0

        ehr = self._ehr_from_row(row)

        # Instance-order shuffle inside the bag if requested (typical MIL augmentation)
        if self.shuffle_instances and feats.ndim >= 2 and feats.shape[0] > 1:
            perm = torch.randperm(feats.shape[0])
            feats = feats[perm]
            if coords_t is not None and coords_t.ndim == 2 and coords_t.shape[0] == perm.shape[0]:
                coords_t = coords_t[perm]

        # Concat mode: append the EHR values as extra instance-level channels (kept for compatibility)
        if self.concat and ehr.numel() > 0 and feats.ndim == 2:
            # feats: (N, D), ehr: (K,) -> tile ehr to (N, K) and cat along feature dim
            ehr_tiled = ehr.float().unsqueeze(0).repeat(feats.shape[0], 1).double()
            feats = torch.cat([feats, ehr_tiled], dim=1)

        # Return signature preserved from your original code:
        #  train: (features, ehr), (label, time_to_event, censorship)
        #  test : (features, ehr, coords), (label, time_to_event, censorship), (patient_id)
        label_t = torch.tensor(label, dtype=torch.int64)
        tte_t = torch.tensor([time_to_event], dtype=torch.float64)
        cens_t = torch.tensor([censorship], dtype=torch.float64)

        if self.state == "test":
            patient = bag_stem
            return (feats, ehr.float(), coords_t), (label_t, tte_t, cens_t), (patient)
        else:
            return (feats, ehr.float()), (label_t, tte_t, cens_t)
