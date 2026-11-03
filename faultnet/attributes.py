"""Seismic/potential-field attribute stack and augmentation for GEMS.

What transfers from seismic practice, and what does not
-------------------------------------------------------
GEMS is gridded potential-field geophysics (magnetics, gravity, elevation,
geodetics), not seismic wiggle traces, so the transfer has to be selective.

Transfers directly:
  * **Coherence / semblance.** The classic seismic fault attribute. A fault is a
    lateral discontinuity, and the structure tensor detects exactly that.
  * **AGC (automatic gain control).** Sliding-window RMS balancing. Standard
    first step on any seismic line; here it stops strong regional anomalies
    from swamping weak local fault signatures.
  * **Instantaneous / analytic-signal attributes.** In potential fields the
    analytic signal `sqrt(HG^2 + VG^2)` and the tilt derivative
    `atan2(VG, HG)` are the standard edge detectors, and the tilt derivative
    in particular is amplitude-independent, so it marks weak and strong edges
    alike.
  * **Elastic/structural warping, gain jitter, additive noise.** Used in
    seismic fault-detection training (FaultSeg3D and successors).

Does not transfer:
  * Bandpass filtering and spectral whitening: there is no time axis.
  * Polarity reversal: valid for a seismic wavelet, but `tmi`, `rtp` and
    `depth_to_base_surf` are absolute physical quantities whose sign is not
    arbitrary. Not applied.
  * Trace dropout: no trace geometry. The band-dropout analogue is offered
    instead.

Note on rotation: `DARL_TTA.md` found rotation/reflection TTA *hurt* by
5.6-7%, because fault orientation in the Basin and Range is strongly preferred
(NNE) and the directional bands are not transform-invariant. The sampler here
therefore makes rotation optional so that prior can be preserved.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

BAND_INDEX = {
    "mag_anom": 0, "rtp": 1, "tmi_hg": 2, "geod_2ndinv": 3,
    "iso_grav_anom_slope": 4, "tc": 5, "geod_shearrate": 6,
    "geod_dilaterate": 7, "tmi_vg": 8, "deq_n100a15": 9,
    "iso_grav_anom_vg": 10, "det_elev": 11, "iso_grav_anom": 12, "tmi": 13,
    "depth_to_base_surf": 14, "ieq_n100a15": 15, "cond_surf": 16,
    "iso_grav_anom_hg": 17, "det_elev_slope": 18,
}


def _robust(a, valid, lo=1.0, hi=99.0):
    v = a[valid]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(a)
    p1, p99 = np.percentile(v, [lo, hi])
    rng = max(p99 - p1, 1e-6)
    return np.clip((a - p1) / rng, -2.0, 2.0).astype(np.float32)


def structure_coherence(field, sigma_grad=1.0, sigma_tensor=3.0):
    """Structure-tensor anisotropy: the potential-field analogue of seismic
    coherence. High on linear discontinuities such as fault traces."""
    f = ndimage.gaussian_filter(field.astype(np.float32), sigma_grad)
    gy, gx = np.gradient(f)
    jxx = ndimage.gaussian_filter(gx * gx, sigma_tensor)
    jyy = ndimage.gaussian_filter(gy * gy, sigma_tensor)
    jxy = ndimage.gaussian_filter(gx * gy, sigma_tensor)
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = np.sqrt(np.maximum(tr * tr - 4 * det, 0.0))
    return (disc / (tr + 1e-9)).astype(np.float32)


def agc(field, window=31, eps=1e-6):
    """Automatic gain control: divide by local RMS, as on a seismic line."""
    f = np.nan_to_num(field.astype(np.float32))
    k = np.ones((window, window), np.float32) / (window * window)
    rms = np.sqrt(np.maximum(ndimage.convolve(f * f, k, mode="nearest"), 0.0))
    return (f / (rms + eps)).astype(np.float32)


def tilt_derivative(vg, hg):
    """atan2(vertical gradient, horizontal gradient). Amplitude-independent,
    so weak edges register as strongly as large anomalies."""
    return np.arctan2(vg, np.abs(hg) + 1e-9).astype(np.float32)


def analytic_signal(vg, hg):
    return np.sqrt(vg * vg + hg * hg).astype(np.float32)


def build_attributes(x, valid):
    """Return (H, W, K) extra channels appended to the 19 raw bands.

    `x` is the *raw* (unnormalised) band stack of shape (H, W, 19).
    """
    b = {name: x[..., i] for name, i in BAND_INDEX.items()}
    out, names = [], []

    def add(name, arr):
        out.append(_robust(np.nan_to_num(arr), valid))
        names.append(name)

    # Potential-field edge detectors built from the supplied gradients.
    add("tilt_tmi", tilt_derivative(b["tmi_vg"], b["tmi_hg"]))
    add("tilt_grav", tilt_derivative(b["iso_grav_anom_vg"], b["iso_grav_anom_hg"]))
    add("as_tmi", analytic_signal(b["tmi_vg"], b["tmi_hg"]))
    add("as_grav", analytic_signal(b["iso_grav_anom_vg"], b["iso_grav_anom_hg"]))

    # Seismic-style coherence on the three most structurally informative fields.
    add("coh_rtp", structure_coherence(np.nan_to_num(b["rtp"])))
    add("coh_grav", structure_coherence(np.nan_to_num(b["iso_grav_anom"])))
    add("coh_elev", structure_coherence(np.nan_to_num(b["det_elev"])))
    add("coh_base", structure_coherence(np.nan_to_num(b["depth_to_base_surf"])))

    # AGC-balanced magnetics and gravity.
    add("agc_rtp", agc(np.nan_to_num(b["rtp"])))
    add("agc_grav", agc(np.nan_to_num(b["iso_grav_anom"])))

    stack = np.stack(out, axis=-1).astype(np.float32)
    stack[~valid] = 0.0
    return stack, names


def build_context(x, valid, sigmas=(8.0, 32.0),
                  bands=("rtp", "iso_grav_anom", "det_elev", "depth_to_base_surf",
                         "tmi_hg", "iso_grav_anom_hg")):
    """Regional-context channels: the key bands smoothed at coarse scales.

    A 64- or 128-px window cannot see the kilometre-scale continuity that
    distinguishes a fault from local texture. Rather than enlarge the window
    (128 px gave only a small gain), append each pixel's regional trend at
    sigma 8 px (~0.8 km) and 32 px (~3.2 km) as extra input channels. The
    network then has local detail and regional context at native resolution,
    with no change to the sampler or the sliding-window inference.
    """
    out, names = [], []
    for name in bands:
        f = np.nan_to_num(x[..., BAND_INDEX[name]]).astype(np.float32)
        f[~valid] = 0.0
        for s in sigmas:
            g = ndimage.gaussian_filter(f, s)
            out.append(_robust(g, valid)); names.append(f"ctx{int(s)}_{name}")
    stack = np.stack(out, axis=-1).astype(np.float32)
    stack[~valid] = 0.0
    return stack, names


# ---------------------------------------------------------------------------
# augmentation
# ---------------------------------------------------------------------------
def elastic_warp(patches, rng, alpha=6.0, sigma=8.0):
    """Smooth random deformation. Standard in seismic fault-detection training;
    it mimics structural variation without inventing new geology."""
    x, y, m = patches
    h, w = y.shape
    dx = ndimage.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma) * alpha
    dy = ndimage.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma) * alpha
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.stack([yy + dy, xx + dx])
    xo = np.stack([ndimage.map_coordinates(x[..., c], coords, order=1, mode="nearest")
                   for c in range(x.shape[-1])], axis=-1)
    yo = ndimage.map_coordinates(y, coords, order=1, mode="nearest")
    mo = ndimage.map_coordinates(m, coords, order=0, mode="nearest")
    return xo.astype(np.float32), yo.astype(np.float32), mo.astype(np.float32)


class SeismicAugPatchSampler:
    """Patch sampler with seismic-style augmentation.

    `rotate=False` preserves the regional fault-orientation prior, which the
    TTA experiment in DARL_TTA.md suggests matters here.
    """

    def __init__(self, x, faults, target, valid, patch, seed, mode,
                 rotate=True, flip=True, gain=0.0, noise=0.0,
                 elastic=0.0, band_dropout=0.0, hard_neg=None, hard_frac=0.25,
                 fold=(0.40, 0.60)):
        """
        hard_neg : optional bool mask of confident false-positive pixels from a
                   previous model. When given, `hard_frac` of each batch is
                   centred on those pixels so the network is shown exactly where
                   it was wrong. This targets efficiency (TP per unit mass),
                   which is the quantity the metric is saturating on.
        fold     : (lo, hi) longitude fractions of the held-out strip. Models
                   trained on different folds have seen different data, which
                   gives ensemble diversity that seed changes do not.
        """
        self.x, self.faults, self.target, self.valid = x, faults, target, valid
        self.patch = patch
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.rotate, self.flip = rotate, flip
        self.gain, self.noise = gain, noise
        self.elastic, self.band_dropout = elastic, band_dropout
        self.hard_frac = hard_frac if hard_neg is not None else 0.0
        h, w = faults.shape
        self.lo, self.hi = int(fold[0] * w), int(fold[1] * w)

        def region_keep(xx):
            if mode == "all":
                return np.ones_like(xx, bool)
            if mode == "validation":
                return (xx >= self.lo + patch // 2) & (xx < self.hi - patch // 2)
            return (xx < self.lo - patch // 2) | (xx >= self.hi + patch // 2)

        yy, xx = np.nonzero(faults)
        keep = region_keep(xx)
        self.pos = np.column_stack([yy[keep], xx[keep]])
        self.hard = None
        if hard_neg is not None:
            hy, hx = np.nonzero(hard_neg)
            hk = region_keep(hx)
            if hk.sum() > 0:
                self.hard = np.column_stack([hy[hk], hx[hk]])

    def one(self, positive, hard=False):
        h, w = self.faults.shape
        p = self.patch
        for _ in range(500):
            if hard and self.hard is not None:
                cy, cx = self.hard[self.rng.integers(len(self.hard))]
                y0 = int(np.clip(cy - self.rng.integers(p // 4, 3 * p // 4), 0, h - p))
                x0 = int(np.clip(cx - self.rng.integers(p // 4, 3 * p // 4), 0, w - p))
            elif positive:
                cy, cx = self.pos[self.rng.integers(len(self.pos))]
                y0 = int(np.clip(cy - self.rng.integers(p // 4, 3 * p // 4), 0, h - p))
                x0 = int(np.clip(cx - self.rng.integers(p // 4, 3 * p // 4), 0, w - p))
            else:
                y0 = int(self.rng.integers(0, h - p + 1))
                x0 = int(self.rng.integers(0, w - p + 1))
            c = x0 + p // 2
            ok = (self.mode == "all"
                  or (self.mode == "validation" and self.lo <= c < self.hi)
                  or (self.mode == "train" and (c < self.lo - p // 2 or c >= self.hi + p // 2)))
            if ok and self.valid[y0:y0 + p, x0:x0 + p].mean() > 0.65:
                return y0, x0
        raise RuntimeError("Unable to sample patch")

    def batch(self, n):
        xb, yb, mb = [], [], []
        p = self.patch
        n_hard = int(round(n * self.hard_frac)) if self.hard is not None else 0
        n_pos = max(1, 3 * (n - n_hard) // 4)
        for i in range(n):
            if i < n_hard:
                y0, x0 = self.one(False, hard=True)
            else:
                y0, x0 = self.one((i - n_hard) < n_pos)
            x = self.x[y0:y0 + p, x0:x0 + p].copy()
            y = self.target[y0:y0 + p, x0:x0 + p].copy()
            m = self.valid[y0:y0 + p, x0:x0 + p].astype(np.float32)

            if self.rotate:
                k = int(self.rng.integers(4))
                x, y, m = np.rot90(x, k), np.rot90(y, k), np.rot90(m, k)
            if self.flip:
                if self.rng.random() < 0.5:
                    x, y, m = x[:, ::-1], y[:, ::-1], m[:, ::-1]
                if self.rng.random() < 0.5:
                    x, y, m = x[::-1], y[::-1], m[::-1]
            x = np.ascontiguousarray(x); y = np.ascontiguousarray(y); m = np.ascontiguousarray(m)

            if self.elastic and self.rng.random() < self.elastic:
                x, y, m = elastic_warp((x, y, m), self.rng)
            if self.gain:
                g = (1.0 + self.gain * self.rng.standard_normal(x.shape[-1])).astype(np.float32)
                x = x * g
            if self.noise:
                x = x + self.noise * self.rng.standard_normal(x.shape).astype(np.float32)
            if self.band_dropout and self.rng.random() < self.band_dropout:
                c = int(self.rng.integers(x.shape[-1] - 1))  # never drop the valid-mask channel
                x[..., c] = 0.0

            xb.append(np.ascontiguousarray(np.moveaxis(x, -1, 0)))
            yb.append(y); mb.append(m)
        return np.stack(xb), np.stack(yb), np.stack(mb)


# ---------------------------------------------------------------------------
# known faults as input: masked-fault training
# ---------------------------------------------------------------------------
def box_channel(proposals, pad=3):
    """Inference-time box prompts: the padded bounding box of every connected
    component of a proposal map (e.g. an earlier fault map)."""
    lab, n = ndimage.label(proposals > 0, structure=np.ones((3, 3)))
    box = np.zeros(proposals.shape, np.float32)
    H, W = proposals.shape
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        box[max(0, sl[0].start - pad):min(H, sl[0].stop + pad), max(0, sl[1].start - pad):min(W, sl[1].stop + pad)] = 1.0
    return box[..., None]


def fault_channels(visible, scale=8.0):
    """Two input channels from a visible-fault mask: the mask itself and
    exp(-distance / scale), which tells the network how far each pixel is from
    the nearest listed fault."""
    d = ndimage.distance_transform_edt(~visible).astype(np.float32)
    return np.stack([visible.astype(np.float32), np.exp(-d / scale)], axis=-1)


class FaultInputSampler(SeismicAugPatchSampler):
    """Hidden-fault recovery inside every training patch.

    At inference the known faults are given and the unlisted ones are sought. Here,
    per patch, a random share of the fault strands (connected components) is
    hidden. The visible strands become input channels; the hidden strands, plus
    any pseudo-labels (unlisted by construction), become the target; pixels
    within `buffer` px of visible strands are removed from the loss, because the
    task is to find faults that are not yet listed. Distances are computed on a padded window so
    strands just outside the patch still count as context.
    """

    def __init__(self, x, faults, valid, patch, seed, mode, extra_pos=None,
                 hide=(0.2, 0.5), pad=24, buffer=2.0, box_prompt=False, **kw):
        super().__init__(x, faults, faults.astype(np.float32), valid, patch, seed, mode, **kw)
        self.lab, _ = ndimage.label(faults, structure=np.ones((3, 3)))
        self.extra = extra_pos if extra_pos is not None else np.zeros_like(faults)
        self.hide, self.pad, self.buffer = hide, pad, buffer
        self.box_prompt = box_prompt

    def boxes_for(self, target, rng):
        """YoloSAM-style box prompts: one jittered, size-augmented bounding box
        per target component (prob 0.8), plus 1-3 distractor boxes with no
        target inside, so the segmenter learns to reject false proposals.
        Returns the box channel and the loss mask restricted to boxes."""
        lab, n = ndimage.label(target > 0, structure=np.ones((3, 3)))
        box = np.zeros(target.shape, np.float32); H, W = target.shape
        for i in range(1, n + 1):
            if rng.random() > 0.8:
                continue
            ys, xs = np.where(lab == i)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            hh, ww = max(y1 - y0 + 1, 4) * rng.uniform(1.0, 1.3) / 2 + 2, max(x1 - x0 + 1, 4) * rng.uniform(1.0, 1.3) / 2 + 2
            oy, ox = rng.integers(-3, 4), rng.integers(-3, 4)
            box[max(0, int(cy + oy - hh)):min(H, int(cy + oy + hh) + 1),
                max(0, int(cx + ox - ww)):min(W, int(cx + ox + ww) + 1)] = 1.0
        for _ in range(rng.integers(1, 4)):
            s = rng.integers(6, 24); y0 = rng.integers(0, max(1, H - s)); x0 = rng.integers(0, max(1, W - s))
            if target[y0:y0 + s, x0:x0 + s].sum() == 0:
                box[y0:y0 + s, x0:x0 + s] = 1.0
        return box

    def batch(self, n):
        xb, yb, mb = [], [], []
        p, P = self.patch, self.pad
        H, W = self.faults.shape
        for i in range(n):
            y0, x0 = self.one(i < max(1, 3 * n // 4))
            ya, xa = max(0, y0 - P), max(0, x0 - P)
            yb_, xb_ = min(H, y0 + p + P), min(W, x0 + p + P)
            lab = self.lab[ya:yb_, xa:xb_]
            ids = np.unique(lab[lab > 0])
            frac = self.rng.uniform(*self.hide)
            hidden_ids = ids[self.rng.random(len(ids)) < frac]
            if len(ids) and not len(hidden_ids):
                hidden_ids = ids[[self.rng.integers(len(ids))]]
            hid = np.isin(lab, hidden_ids) & (lab > 0)
            vis = (lab > 0) & ~hid
            ch = fault_channels(vis)
            dvis = ndimage.distance_transform_edt(~vis)
            sy, sx = y0 - ya, x0 - xa
            win = (slice(sy, sy + p), slice(sx, sx + p))
            x = np.concatenate([self.x[y0:y0 + p, x0:x0 + p], ch[win]], -1)
            y = (hid[win] | self.extra[y0:y0 + p, x0:x0 + p]).astype(np.float32)
            m = (self.valid[y0:y0 + p, x0:x0 + p] & (dvis[win] > self.buffer)).astype(np.float32)
            if self.box_prompt:
                box = self.boxes_for(y, self.rng)
                x = np.concatenate([x, box[..., None]], -1)
                m = m * box                      # learn only inside proposals
            if self.rotate:
                k = int(self.rng.integers(4))
                x, y, m = np.rot90(x, k), np.rot90(y, k), np.rot90(m, k)
            if self.flip:
                if self.rng.random() < 0.5:
                    x, y, m = x[:, ::-1], y[:, ::-1], m[:, ::-1]
                if self.rng.random() < 0.5:
                    x, y, m = x[::-1], y[::-1], m[::-1]
            xb.append(np.ascontiguousarray(np.moveaxis(x, -1, 0)))
            yb.append(np.ascontiguousarray(y)); mb.append(np.ascontiguousarray(m))
        return np.stack(xb), np.stack(yb), np.stack(mb)
