import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("reference_dir", type=Path)
parser.add_argument("mlx_dir", type=Path)
arguments = parser.parse_args()

reference_files = sorted(arguments.reference_dir.glob("*.npy"))
assert reference_files
for reference_path in reference_files:
    mlx_path = arguments.mlx_dir / reference_path.name
    assert mlx_path.exists(), mlx_path
    reference = np.load(reference_path)
    mlx = np.load(mlx_path)
    assert reference.shape == mlx.shape, (
        reference_path.name,
        reference.shape,
        mlx.shape,
    )
    difference = np.abs(reference - mlx)
    scale = np.maximum(np.abs(reference), 1e-7)
    print(
        f"{reference_path.stem:12s} shape={str(reference.shape):24s} "
        f"max_abs={difference.max():.9g} "
        f"mean_abs={difference.mean():.9g} "
        f"max_rel={np.max(difference / scale):.9g}"
    )
