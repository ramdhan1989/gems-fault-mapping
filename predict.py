"""Run a trained checkpoint over the full grid and write the thinned probability map.

usage: python predict.py --checkpoint outputs/res34_model.pt --data data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from faultnet.data import load_inputs
from faultnet.infer import predict_thinned, write_prob
from faultnet.model import Wrapped

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=HERE / "outputs" / "res34_model.pt")
    p.add_argument("--data", type=Path, default=HERE / "data")
    p.add_argument("--out", type=Path, default=HERE / "outputs" / "res34_thinned.tif")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    seed = int(ck.get("record", {}).get("seed", 20260922))   # the input normalisation sample depends on the seed
    labels = a.data / "existing_faults.tif"
    image, faults, valid, *_ = load_inputs(a.data / "gems-geodawn-numerical-features.tif", labels, seed,
                                           cache=a.out.parent / "attribute_stack.npy")
    gen = Wrapped(ck["channels"]); gen.load_state_dict(ck["model"]); gen.to(a.device).eval()
    thin, _ = predict_thinned(gen, image, faults, valid, torch.device(a.device))
    write_prob(a.out, thin, faults, valid, labels, ck.get("record"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
