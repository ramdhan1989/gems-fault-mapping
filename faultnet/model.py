"""SE-ResNet-34 U-Net with feature-pyramid attention (DeepSeismic Res34Unetv4).

- the FPA block is made size-based (the original assumes a fixed x16 pooling size),
- the first convolution is widened from 3 to `channels` inputs, initialised with the
  mean of the ImageNet RGB filters (scaled so the response magnitude is unchanged),
- the network outputs one logit map; Wrapped turns it into a probability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import deepseismic_resnet_unet as ru


def _widen(conv: nn.Conv2d, channels: int) -> nn.Conv2d:
    new = nn.Conv2d(channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        w = conv.weight.mean(1, keepdim=True).repeat(1, channels, 1, 1) * (conv.in_channels / channels)
        new.weight.copy_(w)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _fpa_forward(self, x):                      # size-based instead of fixed x16
    x_glob = self.glob(x)
    x_glob = F.interpolate(x_glob, size=x.shape[-2:], mode="bilinear", align_corners=True)
    d2 = self.down2_1(x); d3 = self.down3_1(d2)
    d2 = self.down2_2(d2); d3 = self.down3_2(d3)
    d3 = F.interpolate(d3, size=d2.shape[-2:], mode="bilinear", align_corners=True)
    d2 = d2 + d3
    d2 = F.interpolate(d2, size=x.shape[-2:], mode="bilinear", align_corners=True)
    return self.conv1(x) * d2 + x_glob


def build_res34unet(channels: int) -> nn.Module:
    ru.FPAv2.forward = _fpa_forward
    net = ru.Res34Unetv4(n_classes=1)             # downloads ImageNet ResNet-34 weights on first use
    net.conv1[0] = _widen(net.conv1[0], channels)
    return net


class Wrapped(nn.Module):
    """Returns (probability, None). The optional t / noise arguments are accepted and ignored."""

    is_darl = False

    def __init__(self, channels: int):
        super().__init__()
        self.net = build_res34unet(channels)

    def forward(self, x, t=None, noise=None):
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 4:
            out = out[:, 0]
        return torch.sigmoid(out.float()).clamp(1e-4, 1 - 1e-4), None

    @torch.no_grad()
    def predict(self, x):
        p, _ = self(x)
        return p
