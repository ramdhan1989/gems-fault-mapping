"""Shared helpers: raster loading, sliding-window inference, ridge thinning,
the distance-weighted Tversky evaluation metric, and the 1-D profile critic.
"""
from __future__ import annotations

import numpy as np
import rasterio
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

from .losses import OFFSETS
from .mda_gan_dnet import spectral_norm

def load_rasters(a):
    with rasterio.open(a.labels) as d:
        labels = d.read(1); transform, crs = d.transform, d.crs
    valid = labels >= 0; faults = labels == 1
    with rasterio.open(a.features) as d:
        if d.shape != labels.shape or d.transform != transform or d.crs != crs:
            raise ValueError("Feature and label rasters are not aligned")
        x = np.moveaxis(d.read(out_dtype="float32"), 0, -1); nodata = d.nodata
        bands = [d.tags(i).get("band_name", f"band_{i}") for i in range(1, d.count+1)]
    bad = ~np.isfinite(x)
    if nodata is not None and np.isfinite(nodata): bad |= x == nodata
    x[bad] = np.nan
    rng = np.random.default_rng(a.seed)
    idx = np.flatnonzero(valid.ravel()); idx = rng.choice(idx, min(500_000, len(idx)), replace=False)
    sample = x.reshape(-1, x.shape[-1])[idx]
    med = np.nanmedian(sample, 0); q25, q75 = np.nanpercentile(sample, [25, 75], axis=0)
    scale = np.maximum(q75-q25, 1e-6)
    x = np.nan_to_num(np.clip((x-med)/scale, -8, 8), nan=0, posinf=8, neginf=-8).astype("float32")
    x[~valid] = 0
    # The validity mask is an image channel so the network can distinguish the survey edge.
    x = np.concatenate([x, valid[..., None].astype("float32")], axis=-1)
    dist = distance_transform_edt(~faults)
    soft = np.maximum(1 - dist/3.0, 0).astype("float32") * valid
    return x, faults, soft, valid, med, scale, bands


@torch.no_grad()
def infer(gen, image, valid, patch, stride, device, batch_size=12, offsets=((0, 0),)):
    """Sliding-window inference, averaged over several shifted grids.

    Multiple offsets suppress the stride-periodic seam that is 33-169x the
    median spectral power of a single-grid map.
    """
    gen.eval()
    h, w, _ = image.shape
    total = np.zeros((h, w), np.float32)
    weights = np.zeros((h, w), np.float32)
    axis = np.maximum(np.hanning(patch), 0.1).astype(np.float32)
    window = axis[:, None] * axis[None, :]

    for oy, ox in offsets:
        ys = list(range(oy, h - patch + 1, stride)) or [0]
        xs = list(range(ox, w - patch + 1, stride)) or [0]
        if ys[-1] != h - patch:
            ys.append(h - patch)
        if xs[-1] != w - patch:
            xs.append(w - patch)
        coords = [(y, x) for y in ys for x in xs if valid[y:y+patch, x:x+patch].any()]
        for s in range(0, len(coords), batch_size):
            cc = coords[s:s+batch_size]
            patches = np.stack([np.moveaxis(image[y:y+patch, x:x+patch], -1, 0) for y, x in cc])
            pred = gen.predict(torch.from_numpy(patches).to(device)).cpu().numpy()
            for (y, x), p in zip(cc, pred):
                total[y:y+patch, x:x+patch] += p * window
                weights[y:y+patch, x:x+patch] += window
            if s % (batch_size * 40) == 0:
                print(f"  inference offset({oy},{ox}) {min(s+len(cc), len(coords))}/{len(coords)}",
                      flush=True)
    out = total / np.maximum(weights, 1e-8)
    out[~valid] = 0
    return out


def ridge_thin(p, sigma=1.6, gamma=0.4):
    """Tuned ridge NMS + sharpening, the post-processing validated separately."""
    from scipy import ndimage
    s = ndimage.gaussian_filter(p.astype(np.float32), sigma)
    gyy = ndimage.gaussian_filter(s, sigma, order=(2, 0))
    gxx = ndimage.gaussian_filter(s, sigma, order=(0, 2))
    gxy = ndimage.gaussian_filter(s, sigma, order=(1, 1))
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    nx, ny = np.cos(theta), np.sin(theta)
    h, w = p.shape
    yy, xx = np.mgrid[0:h, 0:w]
    crest = np.ones(p.shape, bool)
    for sgn in (1.0, -1.0):
        sy = np.clip(np.round(yy + sgn * ny).astype(np.int32), 0, h - 1)
        sx = np.clip(np.round(xx + sgn * nx).astype(np.int32), 0, w - 1)
        crest &= p >= p[sy, sx] - 1e-6
    out = np.where(crest, p, 0.0)
    return np.where(out >= 0.02, np.power(np.clip(out, 0, 1), gamma), 0.0).astype(np.float32)


def strip_mask(valid, faults, frac_lo=0.40, frac_hi=0.60, buffer_px=3.0):
    """Central-longitude strip the trainer never samples, with a known-fault buffer."""
    from scipy.ndimage import distance_transform_edt
    w = valid.shape[1]
    lo, hi = int(frac_lo * w), int(frac_hi * w)
    strip = np.zeros(valid.shape, bool)
    strip[:, lo:hi] = True
    outside = faults.copy()
    outside[:, lo:hi] = False
    far = distance_transform_edt(~outside) > buffer_px
    return valid & strip & far, lo, hi


def exact_metric(pred, truth, mask, alpha=0.2, beta=0.8, radius=3.0):
    from scipy.ndimage import distance_transform_edt
    pred = np.where(mask & np.isfinite(pred), pred, 0.0).astype(np.float32)
    truth = truth & mask
    if truth.any():
        fp_w = np.minimum(distance_transform_edt(~truth) / radius, 1.0)
    else:
        fp_w = np.ones(pred.shape, np.float32)
    fp = float((pred * fp_w).sum(dtype=np.float64))
    best = np.zeros(pred.shape, np.float32)
    h, w = pred.shape
    for dy, dx, wt in OFFSETS:
        dy0, dy1 = max(0, dy), min(h, h + dy)
        dx0, dx1 = max(0, dx), min(w, w + dx)
        sy0, sy1 = max(0, -dy), min(h, h - dy)
        sx0, sx1 = max(0, -dx), min(w, w - dx)
        np.maximum(best[dy0:dy1, dx0:dx1], pred[sy0:sy1, sx0:sx1] * wt,
                   out=best[dy0:dy1, dx0:dx1])
    tp = float(best[truth].sum(dtype=np.float64))
    fn = float(truth.sum() - tp)
    return tp / (tp + alpha * fp + beta * fn + 1e-8)


def profiles(mask, n_lines: int, generator: torch.Generator | None = None):
    """Sample row and column profiles, mirroring MDA-GAN's get_2d_slice."""
    b, _, h, w = mask.shape
    ri = torch.randint(0, h, (min(n_lines, h),), device=mask.device, generator=generator)
    ci = torch.randint(0, w, (min(n_lines, w),), device=mask.device, generator=generator)
    rows = mask[:, :, ri, :].permute(0, 2, 1, 3).reshape(-1, 1, w)
    cols = mask[:, :, :, ci].permute(0, 3, 1, 2).reshape(-1, 1, h)
    return rows, cols


class D_Net1d(nn.Module):
    """1-D profile critic, the dimensional analogue of MDA-GAN's slice critics.

    Sees a single row or column of the mask as a signal and judges whether its
    cross-sectional shape matches real fault traces.
    """

    def __init__(self, in_channels=1, base=32, use_spectral_norm=True):
        super().__init__()
        b = base
        self.model = nn.Sequential(
            spectral_norm(nn.Conv1d(in_channels, b, 4, 2, 1, bias=not use_spectral_norm),
                          use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(b, b * 2, 4, 2, 1, bias=not use_spectral_norm),
                          use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(b * 2, b * 4, 4, 1, 1, bias=not use_spectral_norm),
                          use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(b * 4, 1, 4, 1, 1, bias=not use_spectral_norm),
                          use_spectral_norm),
        )

    def forward(self, x):
        return self.model(x)

