"""Train the SE-ResNet-34 U-Net fault model (known faults as input) and write its thinned map.

Recipe:
  inputs     19 GeoDAWN bands (robust-normalised) + validity mask + 10 attributes + 2 fault channels
  task       in each 128 px patch 20-50% of the known fault strands are hidden from the input
             and become the targets; confident pseudo-labels (p >= 0.9, > 5 px from known faults)
             are extra positives
  loss       focal Tversky (alpha 0.2, beta 0.8, gamma 0.75) + MDA-GAN style critics
             (2-D patch critic + 1-D row/column profile critics, LSGAN) + TCE loss
  schedule   2,200 steps, batch 8, AdamW 1.5e-4 cosine to 1e-5, seed 20260922
  holdout    the 0.40-0.60 longitude strip is never sampled
Outputs: outputs/res34_model.pt and outputs/res34_thinned.tif (input for postprocess.py).
usage: python train.py --data data
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

import rasterio

from faultnet.attributes import FaultInputSampler, fault_channels
from faultnet.common import D_Net1d, exact_metric, infer, profiles, ridge_thin, strip_mask
from faultnet.data import load_inputs
from faultnet.infer import predict_thinned, write_prob
from faultnet.losses import composite
from faultnet.mda_gan_dnet import AdversarialLoss, D_Net2d, TCE_loss
from faultnet.model import Wrapped

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=HERE / "data")
    p.add_argument("--pseudo", type=Path, default=None,
                   help="optional raster of confident detections from an earlier model, used as extra positives")
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--steps", type=int, default=2200)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--no-eval", action="store_true", help="skip the held-out strip evaluation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    PATCH, STRIDE, BATCH, T_MAX, LINES = 128, 96, 8, 200, 8
    W_TCE, W_ADV, DBASE, FOLD, HIDE = 1.0, 0.30, 32, (0.40, 0.60), (0.2, 0.5)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(a.device)

    feats, labels = a.data / "gems-geodawn-numerical-features.tif", a.data / "existing_faults.tif"
    image, faults, valid, bands, med, scale = load_inputs(feats, labels, a.seed, cache=a.out / "attribute_stack.npy")

    # optional pseudo-labels: confident earlier detections away from known faults, never in the holdout strip
    pseudo = None
    if a.pseudo is not None:
        with rasterio.open(a.pseudo) as s:
            prev = s.read(1).astype(np.float32)
        prev = np.where(np.isfinite(prev) & valid, prev, 0.0)
        w_ = valid.shape[1]; lo_, hi_ = int(FOLD[0] * w_), int(FOLD[1] * w_)
        far = ndimage.distance_transform_edt(~faults) > 5.0
        pseudo = (prev >= 0.9) & far & valid
        pseudo[:, lo_:hi_] = False
        print(f"pseudo-labels: {int(pseudo.sum()):,} px, holdout strip excluded", flush=True)

    # segmenter + three critics (2-D patch, 1-D rows, 1-D columns)
    in_ch = image.shape[-1] + 2
    gen = Wrapped(in_ch).to(device)
    n_par = sum(q.numel() for q in gen.parameters())
    print(f"[ds_res34unet] channels={in_ch} params={n_par:,}", flush=True)
    d_full = D_Net2d(in_channels=1 + in_ch, base=DBASE).to(device)
    d_row = D_Net1d(1, base=DBASE).to(device)
    d_col = D_Net1d(1, base=DBASE).to(device)
    advloss = AdversarialLoss("lsgan", 1.0, 0.0).to(device)
    tce = TCE_loss().to(device)
    opt_g = torch.optim.AdamW([q for q in gen.parameters() if q.requires_grad], lr=1.5e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_f = torch.optim.Adam(d_full.parameters(), lr=7.5e-5, betas=(0.5, 0.999))
    opt_r = torch.optim.Adam(d_row.parameters(), lr=7.5e-5, betas=(0.5, 0.999))
    opt_c = torch.optim.Adam(d_col.parameters(), lr=7.5e-5, betas=(0.5, 0.999))
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, max(a.steps, 1), eta_min=1e-5)

    # hidden-fault patches: part of the known strands removed from the input become the targets
    sampler = FaultInputSampler(image, faults, valid, PATCH, a.seed, "train", extra_pos=pseudo, hide=HIDE,
                                fold=FOLD, rotate=True, hard_neg=None, hard_frac=0.25, box_prompt=False)

    # held-out evaluation: in the strip, 30% of strands are hidden from the input and must be recovered,
    # scored away from the visible faults
    eval_mask, _, _ = strip_mask(valid, faults, frac_lo=FOLD[0], frac_hi=FOLD[1])
    lab_all, _ = ndimage.label(faults, structure=np.ones((3, 3)))
    strip_ids = np.unique(lab_all[eval_mask & faults]); strip_ids = strip_ids[strip_ids > 0]
    rng_h = np.random.default_rng(12345)
    hid_ids = strip_ids[rng_h.random(len(strip_ids)) < 0.30]
    hidden_eval = np.isin(lab_all, hid_ids) & faults
    visible_eval = faults & ~hidden_eval
    disc_mask = eval_mask & (ndimage.distance_transform_edt(~visible_eval) > 3.0)
    disc_gt = hidden_eval & disc_mask
    image_eval = np.concatenate([image, fault_channels(visible_eval)], -1).astype(np.float32)

    spec = {"focal_tversky": 1.0}
    cfg = {"pos_weight": 11.0, "focal_alpha": 0.75, "focal_gamma": 2.0, "ft_gamma": 0.75, "ft_alpha": 0.2, "ft_beta": 0.8}
    t0 = time.time()
    for step in range(a.steps):
        xb, yb, mb = sampler.batch(BATCH)
        cond = torch.from_numpy(xb).to(device)
        target = torch.from_numpy(yb).to(device)
        vm = torch.from_numpy(mb).to(device)
        t = torch.randint(0, T_MAX, (cond.shape[0],), device=device)   # not used by this network; drawn so the
        noise = torch.randn_like(cond)                                   # random stream is fixed for a given seed

        prob, _ = gen(cond, t, noise)
        fake = ((prob * 2 - 1)[:, None] * vm[:, None]).float()
        real = (target[:, None] * 2 - 1) * vm[:, None]

        # segmenter step: focal Tversky + TCE + adversarial terms
        opt_g.zero_grad(set_to_none=True)
        fr, fc = profiles(fake, LINES)
        g_full = advloss(d_full(torch.cat([fake, cond], 1)), True, False)
        g_1d = 0.5 * (advloss(d_row(fr), True, False) + advloss(d_col(fc), True, False))
        seg, parts = composite(prob, target, vm, spec, cfg)
        loss = seg + W_TCE * tce(fake, real) + W_ADV * (0.75 * g_full + 0.25 * g_1d)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step+1}")
        loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt_g.step(); sch_g.step()

        # critic steps: real masks vs predicted masks
        fd = fake.detach()
        opt_f.zero_grad(set_to_none=True)
        (0.5 * (advloss(d_full(torch.cat([real, cond], 1)), True, True) +
                advloss(d_full(torch.cat([fd, cond], 1)), False, True))).backward()
        opt_f.step()
        rr, rc = profiles(real, LINES); dr, dc = profiles(fd, LINES)
        opt_r.zero_grad(set_to_none=True)
        (0.5 * (advloss(d_row(rr), True, True) + advloss(d_row(dr), False, True))).backward(); opt_r.step()
        opt_c.zero_grad(set_to_none=True)
        (0.5 * (advloss(d_col(rc), True, True) + advloss(d_col(dc), False, True))).backward(); opt_c.step()
        if step % 400 == 0 or step + 1 == a.steps:
            ps = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
            print(f"  step {step+1}: loss={loss.item():.4f} {ps}  {time.time()-t0:.0f}s", flush=True)

    rec = {"backbone": "ds_res34unet", "params": n_par, "channels": in_ch, "patch": PATCH, "stride": STRIDE,
           "steps": a.steps, "seed": a.seed, "fold": "0.40,0.60", "hide": "0.2,0.5", "objective": "focal_tversky"}
    if not a.no_eval:
        _r = infer(gen, image_eval, valid, PATCH, STRIDE, device); _t = ridge_thin(_r)
        rec.update(holdout_thinned=exact_metric(_t, faults, eval_mask), disc_thinned=exact_metric(_t, disc_gt, disc_mask))
        print(f"  [holdout] thinned={rec['holdout_thinned']:.4f}  hidden-fault recovery={rec['disc_thinned']:.4f}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": gen.state_dict(), "backbone": "ds_res34unet", "channels": in_ch,
                "median": med, "scale": scale, "bands": bands, "record": rec}, a.out / "res34_model.pt")
    thin, _ = predict_thinned(gen, image, faults, valid, device, PATCH, STRIDE)
    write_prob(a.out / "res34_thinned.tif", thin, faults, valid, labels, rec)
    print(f"wrote {a.out / 'res34_model.pt'} and {a.out / 'res34_thinned.tif'} ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
