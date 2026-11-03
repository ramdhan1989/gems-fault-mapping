"""Full-grid inference: 4 shifted sliding-window grids, identity and vertical flip combined by max,
then ridge thinning to 1-px lines. Writes the thinned probability raster."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .common import infer, ridge_thin
from .data import final_image


def predict_thinned(gen, image, faults, valid, device, patch=128, stride=96):
    s = stride
    offsets = ((0, 0), (s // 4, s // 4), (s // 2, s // 2), (3 * s // 4, 3 * s // 4))
    img = final_image(image, faults)
    raw = infer(gen, img, valid, patch, stride, device, offsets=offsets)
    # fault strike is not flip-symmetric: max keeps the stronger evidence of either view
    img_v = np.ascontiguousarray(img[::-1]); val_v = np.ascontiguousarray(valid[::-1])
    rv = infer(gen, img_v, val_v, patch, stride, device, offsets=offsets)
    raw = np.maximum(raw, np.ascontiguousarray(rv[::-1])).astype(np.float32)
    return ridge_thin(raw), raw


def write_prob(path: Path, thin, faults, valid, template: Path, tags=None):
    """Known-fault pixels set to 0, NaN outside the survey, float32 on the template grid."""
    with rasterio.open(template) as s:
        prof = s.profile.copy()
    prof.update(dtype="float32", count=1, nodata=np.nan, compress="lzw", predictor=3)
    arr = thin.copy(); arr[faults] = 0; arr[~valid] = np.nan
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr.astype(np.float32), 1)
        if tags:
            d.update_tags(**{k: str(v) for k, v in tags.items()})
