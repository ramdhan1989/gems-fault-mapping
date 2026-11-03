"""Binary objectives for GEMS fault segmentation, including focal variants.

The evaluation metric is a distance-weighted Tversky index

    score = TP / (0.2*TP + 0.2*FP + 0.8*N)

with TP credited through a 3-px triangular kernel. Class imbalance is about
1:83.7, so the objective choice matters more than usual.

Available objectives
--------------------
bce             foreground-weighted binary cross entropy (baseline)
focal           focal BCE, Lin et al. 2017
focal_tversky   focal Tversky, Abraham & Khan 2019, alpha/beta set to the metric's
unified_focal   symmetric unified focal, Yeung et al. 2022
metric_tversky  differentiable form of the evaluation metric
cldice          centreline Dice for thin structures

Focal loss is a genuinely reasonable candidate here. It down-weights the easy
background that dominates a 1:83.7 split, so gradient goes to the hard pixels
near fault traces. The caveat is that the metric charges false positives at
only 0.2 while charging misses at 0.8, so an objective that suppresses
confident background can help recall, which is where the remaining gap is.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _offsets(radius: float = 3.0):
    offs = []
    reach = int(np.ceil(radius))
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            w = 1.0 - float(np.hypot(dx, dy)) / radius
            if w > 0:
                offs.append((dy, dx, w))
    return offs


OFFSETS = _offsets(3.0)


def soft_match(p, radius: float = 3.0):
    pad = int(np.ceil(radius))
    q = F.pad(p[:, None], (pad, pad, pad, pad), value=0.0)
    h, w = p.shape[-2:]
    best = torch.zeros_like(p)
    for dy, dx, wt in OFFSETS:
        best = torch.maximum(best, q[:, 0, pad + dy:pad + dy + h, pad + dx:pad + dx + w] * wt)
    return best


# ---------------------------------------------------------------------------
def weighted_bce(prob, target, valid, pos_weight=11.0):
    pr = prob.clamp(1e-4, 1 - 1e-4)
    w = 1 + pos_weight * target
    m = -(target * pr.log() + (1 - target) * (1 - pr).log())
    return (m * w * valid).sum() / (w * valid).sum().clamp_min(1)


def focal_bce(prob, target, valid, alpha=0.75, gamma=2.0):
    """Focal loss (Lin et al. 2017).

    `alpha` weights the rare positive class; `(1-p_t)^gamma` removes the
    contribution of the easy background that makes up 98.8% of pixels.
    """
    pr = prob.clamp(1e-4, 1 - 1e-4)
    pt = target * pr + (1 - target) * (1 - pr)
    at = target * alpha + (1 - target) * (1 - alpha)
    loss = -at * (1 - pt).pow(gamma) * pt.log()
    return (loss * valid).sum() / valid.sum().clamp_min(1)


def focal_tversky(prob, target, valid, alpha=0.2, beta=0.8, gamma=0.75,
                  matched=True):
    """Focal Tversky (Abraham & Khan 2019).

    alpha/beta are set to the metric's own 0.2/0.8. Raising the Tversky
    index to 1/gamma > 1 concentrates learning on hard, low-overlap patches,
    which here means the sparse fault-bearing ones.
    """
    p = prob * valid
    t = target * valid
    dims = (1, 2)
    if matched:
        tp = (soft_match(p) * t).sum(dims)
        n = t.sum(dims)
        fn = (n - tp).clamp_min(0)
    else:
        tp = (p * t).sum(dims)
        fn = ((1 - p) * t).sum(dims)
    fp = (p * (1 - t)).sum(dims)
    ti = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return ((1 - ti).clamp_min(1e-6) ** gamma).mean()


def metric_tversky(prob, target, valid, alpha=0.2, beta=0.8):
    return focal_tversky(prob, target, valid, alpha, beta, gamma=1.0, matched=True)


def unified_focal(prob, target, valid, weight=0.5, delta=0.6, gamma=0.5):
    """Symmetric unified focal loss (Yeung et al. 2022): a focal region term
    combined with a focal distribution term, designed for extreme imbalance."""
    pr = prob.clamp(1e-4, 1 - 1e-4)
    p, t = pr * valid, target * valid
    dims = (1, 2)
    tp = (p * t).sum(dims)
    fp = (p * (1 - t)).sum(dims)
    fn = ((1 - p) * t).sum(dims)
    tversky = (tp + 1.0) / (tp + delta * fn + (1 - delta) * fp + 1.0)
    region = ((1 - tversky).clamp_min(1e-6) ** (1 - gamma)).mean()

    pt = t * pr + (1 - t) * (1 - pr)
    at = t * delta + (1 - t) * (1 - delta)
    dist = (-at * (1 - pt).pow(gamma) * pt.log() * valid).sum() / valid.sum().clamp_min(1)
    return weight * region + (1 - weight) * dist


def _soft_erode(x):
    return torch.min(-F.max_pool2d(-x, (3, 1), 1, (1, 0)),
                     -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def _soft_dilate(x):
    return F.max_pool2d(x, (3, 3), 1, (1, 1))


def soft_skeleton(x, iters=4):
    p1 = _soft_erode(x)
    skel = F.relu(x - _soft_dilate(p1))
    for _ in range(iters):
        x = p1
        p1 = _soft_erode(x)
        d = F.relu(x - _soft_dilate(p1))
        skel = skel + F.relu(d - skel * d)
    return skel


def cl_dice(prob, target, valid, eps=1.0):
    p = (prob * valid)[:, None]
    t = (target * valid)[:, None]
    sp, st = soft_skeleton(p), soft_skeleton(t)
    prec = ((sp * t).sum((1, 2, 3)) + eps) / (sp.sum((1, 2, 3)) + eps)
    sens = ((st * p).sum((1, 2, 3)) + eps) / (st.sum((1, 2, 3)) + eps)
    return (1 - 2.0 * prec * sens / (prec + sens)).mean()


def tversky(prob, target, valid, alpha=0.2, beta=0.8):
    """Plain pixelwise Tversky (Salehi et al. 2017) with the metric's
    alpha/beta: FP weighted 0.2, FN weighted 0.8. No focal exponent, no
    3-px matching."""
    return focal_tversky(prob, target, valid, alpha, beta, gamma=1.0, matched=False)


def sensitivity_specificity(prob, target, valid, r=0.05):
    """Sensitivity-Specificity loss (Brosch et al. 2015).

    L = r * sum((y-p)^2 * y) / sum(y)  +  (1-r) * sum((y-p)^2 * (1-y)) / sum(1-y)

    r weights the sensitivity (fault-pixel) error; the original paper uses a
    small r because the background term is otherwise dominated by the huge
    negative class. With 1:83.7 imbalance here, r is left at the paper's 0.05
    but note that it therefore emphasises specificity, the opposite of what
    the 0.8-weighted-FN metric rewards. Included as asked; expectation is low.
    """
    p, y = prob * valid, target * valid
    err = (y - p) ** 2
    sens = (err * y).sum((1, 2)) / y.sum((1, 2)).clamp_min(1.0)
    spec = (err * (1 - y) * valid).sum((1, 2)) / ((1 - y) * valid).sum((1, 2)).clamp_min(1.0)
    return (r * sens + (1 - r) * spec).mean()


def combo(prob, target, valid, alpha=0.5, ce_ratio=0.5, pos_weight=11.0):
    """Combo loss (Taghanaki et al. 2019): weighted cross entropy + Dice.

    L = ce_ratio * wBCE + (1 - ce_ratio) * (1 - Dice)
    Standard Dice weights FP and FN equally (alpha = 0.5), unlike the metric.
    """
    p, y = prob * valid, target * valid
    pr = prob.clamp(1e-4, 1 - 1e-4)
    w = 1 + pos_weight * target
    bce = -(target * pr.log() + (1 - target) * (1 - pr).log())
    bce = (bce * w * valid).sum() / (w * valid).sum().clamp_min(1)
    dims = (1, 2)
    inter = (p * y).sum(dims)
    dice = (2 * inter + 1.0) / (p.sum(dims) + y.sum(dims) + 1.0)
    return ce_ratio * bce + (1 - ce_ratio) * (1 - dice).mean()


def _lovasz_grad(gt_sorted):
    """Gradient of the Lovász extension of the Jaccard loss (Berman et al. 2018)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(prob, target, valid, tolerance_px=3):
    """Lovász hinge over the 3-px-dilated target: a ranking loss.

    Why here: every backbone reaches ~8-9% efficiency (TP per unit mass) and
    unions cannot raise it, so the bottleneck is how well each model *orders*
    its pixels, which is exactly what thresholding exploits and exactly what
    the Lovász extension optimises. The target is dilated to the metric's
    3-px kernel so a correctly placed 1-px trace is not penalised for being
    one pixel off.
    """
    # Tolerance is applied by IGNORING the 3-px ring around each trace, not by
    # labelling it positive. Dilating the target would teach the network to
    # paint 3-px bands, which is the thickness defect removed earlier.
    tgt = target.float()
    if tolerance_px:
        ring = (soft_match(tgt) > 0) & (tgt == 0)
        valid = valid * (~ring).float()
        # Positive side: a trace pixel is "hit" if the prediction is high
        # anywhere in its kernel, exactly as the metric's max-match credits TP.
        prob = torch.where(tgt > 0, soft_match(prob), prob)
    logits = torch.logit(prob.clamp(1e-4, 1 - 1e-4))
    losses = []
    for lg, tg, vm in zip(logits, tgt, valid):
        m = vm > 0
        lg, tg = lg[m], tg[m]
        if tg.numel() == 0:
            continue
        signs = 2.0 * tg - 1.0
        errors = 1.0 - lg * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        grad = _lovasz_grad(tg[perm])
        losses.append(torch.dot(F.relu(errors_sorted), grad))
    if not losses:
        return prob.sum() * 0.0
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
def kl_smooth(prob, target, valid, eps=0.1, pos_weight=11.0):
    """YoloSAM-style noisy-label objective: KL divergence between the
    prediction and a label-smoothed target (t -> t(1-eps) + eps/2), which is
    BCE against the smoothed target up to a constant; positives weighted."""
    t = target * (1 - eps) + eps / 2
    p = prob.clamp(1e-6, 1 - 1e-6)
    kl = t * torch.log(t / p) + (1 - t) * torch.log((1 - t) / (1 - p))
    w = torch.where(target > 0.5, pos_weight, 1.0) * valid
    return (kl * w).sum() / w.sum().clamp(min=1.0)


def kl_gauss(prob, target, valid, sigma=1.0, pos_weight=11.0):
    """YoloSAM's actual KL term: soft target = Gaussian-blurred mask
    normalised to max 1 (sigma 1 px), KL(target || pred) per pixel."""
    import math
    r = int(math.ceil(3 * sigma)); ax = torch.arange(-r, r + 1, device=prob.device, dtype=torch.float32)
    k = torch.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2)); k = (k / k.sum())[None, None]
    soft = torch.nn.functional.conv2d(target[:, None].float(), k, padding=r)[:, 0]
    mx = soft.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    t = (soft / mx).clamp(1e-6, 1 - 1e-6)
    p = prob.clamp(1e-6, 1 - 1e-6)
    kl = t * torch.log(t / p) + (1 - t) * torch.log((1 - t) / (1 - p))
    w = torch.where(t > 0.05, pos_weight, 1.0) * valid
    return (kl * w).sum() / w.sum().clamp(min=1.0)


def fp_weight(target):
    """The metric's false-positive weight: min(d/3, 1), d = distance to the
    nearest truth pixel. soft_match(target) = max(0, 1 - d/3) exactly."""
    return (1.0 - soft_match(target.float())).clamp(0.0, 1.0)


def exact_metric_loss(prob, target, valid, gamma=1.0, pooled=True):
    """Differentiable form of the evaluation metric with BOTH of its
    mechanics: TP credited through the 3-px triangular kernel on truth
    pixels, FP weighted by distance to truth (near-misses cheap).
    pooled=True forms one ratio over the whole batch, like the metric's
    single ratio over the map; pooled=False averages per-patch ratios."""
    p, t = prob * valid, target * valid
    tp_map = soft_match(p) * t
    fp_map = p * fp_weight(t) * valid
    if pooled:
        tp, fp, n = tp_map.sum(), fp_map.sum(), t.sum()
        s = tp / (0.2 * tp + 0.2 * fp + 0.8 * n + 1.0)
        return (1 - s).clamp_min(1e-6) ** gamma
    dims = (1, 2)
    tp, fp, n = tp_map.sum(dims), fp_map.sum(dims), t.sum(dims)
    s = (tp + 1.0) / (0.2 * tp + 0.2 * fp + 0.8 * n + 1.0)
    return ((1 - s).clamp_min(1e-6) ** gamma).mean()


def focal_tversky_dw(prob, target, valid, alpha=0.2, beta=0.8, gamma=0.75, pooled=False):
    """Focal Tversky with the metric's distance-weighted FP term; optionally
    pooled over the batch."""
    p, t = prob * valid, target * valid
    tp_map = soft_match(p) * t
    fp_map = p * fp_weight(t) * valid
    if pooled:
        tp, fp, n = tp_map.sum(), fp_map.sum(), t.sum()
        ti = (tp + 1.0) / (tp + alpha * fp + beta * (n - tp).clamp_min(0) + 1.0)
        return (1 - ti).clamp_min(1e-6) ** gamma
    dims = (1, 2)
    tp, fp, n = tp_map.sum(dims), fp_map.sum(dims), t.sum(dims)
    ti = (tp + 1.0) / (tp + alpha * fp + beta * (n - tp).clamp_min(0) + 1.0)
    return ((1 - ti).clamp_min(1e-6) ** gamma).mean()


def focal_tversky_pooled(prob, target, valid, alpha=0.2, beta=0.8, gamma=0.75):
    """The focal Tversky (matched TP, unweighted FP) but as one
    ratio over the batch instead of a mean of per-patch ratios."""
    p, t = prob * valid, target * valid
    tp = (soft_match(p) * t).sum(); n = t.sum(); fp = (p * (1 - t)).sum()
    ti = (tp + 1.0) / (tp + alpha * fp + beta * (n - tp).clamp_min(0) + 1.0)
    return (1 - ti).clamp_min(1e-6) ** gamma


def dist_bce(prob, target, valid, pos_weight=11.0):
    """BCE matched to the metric: a truth pixel is satisfied if any
    prediction within the 3-px kernel is high (-log soft_match(p)); a
    negative pixel costs -log(1-p) scaled by its FP weight min(d/3,1)."""
    pr = prob.clamp(1e-4, 1 - 1e-4)
    t = target.float()
    pos = -torch.log(soft_match(pr).clamp(1e-4, 1.0)) * t * pos_weight
    neg = -torch.log(1 - pr) * (1 - t) * fp_weight(t)
    w = (t * pos_weight + (1 - t)) * valid
    return ((pos + neg) * valid).sum() / w.sum().clamp_min(1.0)


def poly1_focal(prob, target, valid, alpha=0.75, gamma=2.0, eps=1.0):
    """Poly-1 focal loss (Leng et al. 2022): focal + eps * (1 - p_t)^(gamma+1)."""
    pr = prob.clamp(1e-4, 1 - 1e-4)
    pt = target * pr + (1 - target) * (1 - pr)
    at = target * alpha + (1 - target) * (1 - alpha)
    loss = -at * (1 - pt).pow(gamma) * pt.log() + eps * (1 - pt).pow(gamma + 1)
    return (loss * valid).sum() / valid.sum().clamp_min(1)


OBJECTIVES = {
    "exact_metric":        lambda p, t, v, c: exact_metric_loss(p, t, v, gamma=c.get("em_gamma", 1.0), pooled=True),
    "exact_metric_focal":  lambda p, t, v, c: exact_metric_loss(p, t, v, gamma=0.75, pooled=True),
    "exact_metric_patch":  lambda p, t, v, c: exact_metric_loss(p, t, v, gamma=0.75, pooled=False),
    "focal_tversky_dw":    lambda p, t, v, c: focal_tversky_dw(p, t, v, gamma=c.get("ft_gamma", 0.75)),
    "focal_tversky_dw_pooled": lambda p, t, v, c: focal_tversky_dw(p, t, v, gamma=c.get("ft_gamma", 0.75), pooled=True),
    "focal_tversky_pooled": lambda p, t, v, c: focal_tversky_pooled(p, t, v, gamma=c.get("ft_gamma", 0.75)),
    "dist_bce":            lambda p, t, v, c: dist_bce(p, t, v, c.get("pos_weight", 11.0)),
    "poly1_focal":         lambda p, t, v, c: poly1_focal(p, t, v),
    "kl_gauss":       lambda p, t, v, c: kl_gauss(p, t, v, c.get("kl_sigma", 1.0), c.get("pos_weight", 11.0)),
    "kl_smooth":      lambda p, t, v, c: kl_smooth(p, t, v, c.get("kl_eps", 0.1), c.get("pos_weight", 11.0)),
    "lovasz":         lambda p, t, v, c: lovasz_hinge(p, t, v),
    "tversky":        lambda p, t, v, c: tversky(p, t, v),
    "sens_spec":      lambda p, t, v, c: sensitivity_specificity(p, t, v, c.get("ss_r", 0.05)),
    "combo":          lambda p, t, v, c: combo(p, t, v, pos_weight=c.get("pos_weight", 11.0)),
    "bce":            lambda p, t, v, c: weighted_bce(p, t, v, c.get("pos_weight", 11.0)),
    "focal":          lambda p, t, v, c: focal_bce(p, t, v, c.get("focal_alpha", 0.75),
                                                   c.get("focal_gamma", 2.0)),
    "focal_tversky":  lambda p, t, v, c: focal_tversky(p, t, v, alpha=c.get("ft_alpha", 0.2),
                                                       beta=c.get("ft_beta", 0.8), gamma=c.get("ft_gamma", 0.75)),
    "unified_focal":  lambda p, t, v, c: unified_focal(p, t, v, gamma=c.get("uf_gamma", 0.5)),
    "metric_tversky": lambda p, t, v, c: metric_tversky(p, t, v),
    "cldice":         lambda p, t, v, c: cl_dice(p, t, v),
}


def composite(prob, target, valid, spec, cfg):
    """`spec` maps objective name -> weight, e.g. {"focal": 0.5, "metric_tversky": 0.5}."""
    total = torch.zeros((), device=prob.device)
    parts = {}
    for name, w in spec.items():
        if w == 0:
            continue
        val = OBJECTIVES[name](prob, target, valid, cfg)
        parts[name] = float(val.detach())
        total = total + w * val
    return total, parts
