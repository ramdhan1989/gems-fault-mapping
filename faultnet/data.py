"""Input stack: 19 robust-normalised GeoDAWN bands + validity mask + 10 derived attributes (30 channels).

The two known-fault channels (mask and exp(-distance/8)) are appended by the fault-input
sampler during training and by `final_image` at inference, giving 32 channels.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio

from .attributes import build_attributes, fault_channels
from .common import load_rasters


def load_inputs(features: Path, labels: Path, seed: int = 20260922, cache: Path | None = None):
    """Returns image (H, W, 30), faults, valid, band names, median, scale."""
    a = SimpleNamespace(features=Path(features), labels=Path(labels), seed=seed)
    image, faults, _soft, valid, med, scale, bands = load_rasters(a)
    if cache is not None and Path(cache).exists():
        extra = np.load(cache)
    else:
        with rasterio.open(features) as d:
            raw = np.moveaxis(d.read(out_dtype="float32"), 0, -1); nod = d.nodata
        bad = ~np.isfinite(raw)
        if nod is not None and np.isfinite(nod):
            bad |= raw == nod
        raw[bad] = np.nan
        extra, _ = build_attributes(raw, valid)
        del raw
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, extra)
    image = np.concatenate([image, extra], axis=-1).astype(np.float32)
    return image, faults, valid, bands, med, scale


def final_image(image, faults):
    """At inference the model sees every known fault."""
    return np.concatenate([image, fault_channels(faults)], -1).astype(np.float32)
