#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python train.py --data data --out outputs
python postprocess.py --data data --prob outputs/res34_thinned.tif --out outputs/fault_map.tif
