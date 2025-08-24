# survivmil/bags_io.py
from __future__ import annotations
import os
from typing import Tuple

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
    # Extract the last two parts, strip 'x' or 'y' and the '.npy' extension if present
    coord_strs = path.split("_")[-2:]
    coords = []
    for s in coord_strs:
        # Remove file extension if present (e.g., 'y12345.npy' -> 'y12345')
        s = s.split(".")[0]
        # Remove leading 'x' or 'y'
        coords.append(int(s[1:]))
    coords = np.array(coords)
    return feats, coords


def load_bag(path: str) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Load a bag of patch features.

    Parameters
    ----------
    path:
        Either a single ``.npy`` file containing all features for a patient or a
        directory of individual patch ``.npy`` files.  In the latter case the
        coordinates are parsed from each filename (``*_x{coord}_y{coord}.npy``).

    Returns
    -------
    features, coords
        ``features`` is ``(N, D)`` and ``coords`` is ``(N, 2)`` or ``None`` if
        coordinates are unavailable.
    """

    if os.path.isdir(path):
        feats_list, coords_list = [], []
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".npy"):
                continue
            f, c = _load_npy(os.path.join(path, fname))
            feats_list.append(f)
            coords_list.append(c)
        if not feats_list:
            raise FileNotFoundError(f"No '.npy' files found in directory '{path}'")
        return np.stack(feats_list), np.stack(coords_list)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return _load_npy(path)
    raise ValueError(f"Unsupported bag extension: {ext}")


def resolve_bag_path(features_dir: str, bag_stem: str) -> str:
    """
    Locate the bag corresponding to ``bag_stem``.

    The bag may either be a directory of ``.npy`` files or a single file with
    extension ``.npy``/``.h5``/``.hdf5``.  The search order prefers directories
    (allowing the UNI embedding layout) and then falls back to single files.
    """

    dir_candidate = os.path.join(features_dir, bag_stem)
    if os.path.isdir(dir_candidate):
        return dir_candidate

    cand = [
        os.path.join(features_dir, f"{bag_stem}.npy"),
        os.path.join(features_dir, f"{bag_stem}.h5"),
        os.path.join(features_dir, f"{bag_stem}.hdf5"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No bag found for stem='{bag_stem}' in {features_dir} (checked directory and .npy/.h5/.hdf5)."
    )
