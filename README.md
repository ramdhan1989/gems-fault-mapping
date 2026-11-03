# Fault mapping from potential-field data with known faults as input

A single segmentation model that maps faults on a 100 m grid from GeoDAWN geophysical and topographic layers. It is given the faults that are already known and learns to find the ones that are missing.

## Algorithm

### 1. Input stack (32 channels)

| channels | content |
|---|---|
| 19 | GeoDAWN feature bands: magnetics, gravity, geodesy, elevation and depth layers. Each is normalised by its median and interquartile range, then clipped to ±8. |
| 1 | validity mask, so the network can recognise the survey edge |
| 10 | derived potential-field attributes: tilt derivatives and analytic signals of magnetics and gravity, structure-tensor coherence of four fields, and automatic gain control of reduced-to-pole magnetics and gravity |
| 2 | known faults: the fault mask, and exp(-distance / 8 px) to the nearest known fault |

### 2. Training task: hidden-fault recovery

In each 128 x 128 training patch, 20-50% of the known fault strands are removed from the two fault channels. The removed strands become the targets. The model therefore learns to infer a fault from the geophysics and from the pattern of the surrounding faults, not to copy the input.

Optionally, `--pseudo` adds confident detections of an earlier model as extra positive labels: p >= 0.9, more than 5 px from a known fault, never in the held-out strip. Patches are augmented with 90-degree rotations and flips.

### 3. Network

An SE-ResNet-34 U-Net with a feature-pyramid attention bridge, about 36M parameters. The encoder starts from ImageNet weights. Its first convolution is widened to 32 inputs, initialised with the mean of the RGB filters. The attention block is made size-independent so it accepts any patch size. The network outputs one probability map.

### 4. Loss

- **Focal Tversky** with alpha 0.2 on false positives, beta 0.8 on false negatives and gamma 0.75. Missing a fault costs four times as much as a false alarm.
- **Adversarial shape critics** in the style of MDA-GAN: a 2-D critic on the predicted patch, and 1-D critics on random rows and columns. Trained with a least-squares GAN loss at weight 0.3, they push predictions toward thin, line-like traces.
- **TCE (tanh cross-entropy) loss** between predicted and true masks, at weight 1.0.

### 5. Optimisation

- 2,200 steps with batch size 8.
- AdamW at 1.5e-4 with cosine decay to 1e-5, and gradient clipping at 1.0.
- The critics use Adam at 7.5e-5.
- Seed 20260922.
- The 0.40-0.60 longitude strip is never sampled. It is used to measure how well hidden faults are recovered.

### 6. Inference

1. Sliding 128 px windows with stride 96 and a Hann weighting, averaged over 4 diagonally shifted grids to remove seams.
2. The identity and vertically flipped views are combined by maximum.
3. Ridge non-maximum suppression along the local normal direction thins the map to 1-px lines, followed by a power-0.4 sharpening.

### 7. Post-processing

1. Probability >= 0.40 becomes 1, everything else 0.
2. Known-fault pixels are set to 0.
3. Detections within 4 px (400 m) of a known fault are removed, because they mostly redraw known faults.
4. Pixels outside the survey area are set to NaN. The output is a float32 GeoTIFF on the input grid (EPSG:32611).

## Usage

```bash
pip install -r requirements.txt
python train.py         # -> outputs/res34_model.pt, outputs/res34_thinned.tif
python postprocess.py   # -> outputs/fault_map.tif
```

To run inference with an existing model:

```bash
python predict.py --checkpoint outputs/res34_model.pt
python postprocess.py
```

`run_all.sh` and `run_all.ps1` run the full pipeline. The input files are listed in `data/README.md`.

## Determinism

GPU convolution kernels are not deterministic, so two runs with the same seed give different line sets. Their quality is the same. In a comparison of two runs, both recovered hidden faults on the held-out strip with the same score (0.158). 77% of one run's pixels lay within 2 px of the other's. Running `predict.py` on the same checkpoint always gives the same map.

## Layout

```
train.py          training, held-out evaluation, full-grid inference
predict.py        inference from a checkpoint
postprocess.py    thresholding, near-fault exclusion, GeoTIFF output and checks
faultnet/
  model.py        SE-ResNet-34 U-Net with feature-pyramid attention, 32 inputs
  data.py         input stack
  attributes.py   derived attributes, hidden-fault patch sampler
  losses.py       focal Tversky and related losses
  infer.py        multi-grid, flip-max inference and raster writing
  common.py       raster loading, sliding windows, ridge thinning, evaluation metric, 1-D critic
  deepseismic_resnet_unet.py   network from microsoft/seismic-deeplearning (MIT)
  mda_gan_dnet.py              2-D critic and GAN/TCE losses from MDA-GAN (Apache-2.0)
data/             input rasters (not included)
third_party/      licenses of the reused code
```

## Third-party code

- `faultnet/deepseismic_resnet_unet.py` comes from [microsoft/seismic-deeplearning](https://github.com/microsoft/seismic-deeplearning), MIT License, unchanged. Its attention block is patched at runtime in `faultnet/model.py`.
- `faultnet/mda_gan_dnet.py` comes from MDA-GAN, Apache License 2.0, unchanged.
- The license texts are in `third_party/`. The ResNet-34 ImageNet weights are downloaded from torchvision on first use.
