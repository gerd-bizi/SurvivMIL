# survivmil/bags_io.py
from __future__ import annotations
import os
import numpy as np

def _load_npy(path: str):
    """
    Loads a single feature vector from a .npy file.
    The coordinates are in the name of the file.
    For example, TMA_1_1_k0188jcx_x39235_y44865.npy
    has coords x39235 and y44865.
    """
    feats = np.load(path, allow_pickle=False)
    # To get coords, we read from file name
    coords = np.array([int(x) for x in path.split("_")[-2:]])
    return feats, coords


def load_bag(path: str):
    """Load a single bag from .npy or .h5/.hdf5. Returns (features, coords_or_None)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return _load_npy(path)
    raise ValueError(f"Unsupported bag extension: {ext}")


def resolve_bag_path(features_dir: str, bag_stem: str):
    """
    Prefer .npy; fallback to .h5/.hdf5 if present.
    Returns absolute path or raises FileNotFoundError.
    """
    cand = [
        os.path.join(features_dir, f"{bag_stem}.npy"),
        os.path.join(features_dir, f"{bag_stem}.h5"),
        os.path.join(features_dir, f"{bag_stem}.hdf5"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No bag found for stem='{bag_stem}' in {features_dir} (tried .npy/.h5/.hdf5).")
