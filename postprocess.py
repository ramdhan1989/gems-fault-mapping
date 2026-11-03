"""Turn the thinned probability map into a binary fault map and check its format.

Steps:
  1. binary: probability >= 0.40 -> 1, else 0
  2. known-fault pixels -> 0
  3. detections within 4 px (400 m) of a known fault -> 0 (they mostly redraw known faults)
  4. NaN outside the valid survey area; float32 on the same grid and CRS as the inputs
usage: python postprocess.py --prob outputs/res34_thinned.tif --data data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prob", type=Path, default=HERE / "outputs" / "res34_thinned.tif")
    p.add_argument("--data", type=Path, default=HERE / "data")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--near", type=float, default=4.0, help="remove detections within this many px of a known fault")
    p.add_argument("--out", type=Path, default=HERE / "outputs" / "fault_map.tif")
    a = p.parse_args()

    with rasterio.open(a.data / "existing_faults.tif") as s:
        lab = s.read(1); prof = s.profile.copy(); shape, crs, tr = s.shape, s.crs, s.transform
    valid, known = lab >= 0, lab == 1
    with rasterio.open(a.prob) as s:
        prob = np.nan_to_num(s.read(1).astype(np.float32))

    dk = ndimage.distance_transform_edt(~known)
    pred = (prob >= a.threshold) & valid & ~known & (dk > a.near)
    arr = pred.astype(np.float32); arr[~valid] = np.nan
    prof.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(a.out, "w", **prof) as d:
        d.write(arr, 1)

    # format checks
    with rasterio.open(a.out) as s:
        b = s.read(1); ok = {
            "shape": s.shape == shape, "crs": s.crs == crs, "transform": s.transform == tr,
            "float32": s.dtypes[0] == "float32", "nan outside survey": bool(np.isnan(b[~valid]).all()),
            "finite inside": bool(np.isfinite(b[valid]).all()), "binary 0/1": set(np.unique(b[valid])) <= {0.0, 1.0},
            "known faults 0": bool((b[known] == 0).all())}
    for k, v in ok.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    n_all = int(((prob >= a.threshold) & valid & ~known).sum())
    print(f"fault pixels: {int(pred.sum()):,} (removed {n_all - int(pred.sum()):,} within {a.near:g} px of known faults)")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
