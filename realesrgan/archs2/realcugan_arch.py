"""
RealCUGAN (Real Cascaded U-Net for anime upscaling), self-contained.

Covers checkpoints from the bilibili/ailab Real-CUGAN project
(up2x/up3x/up4x, "-denoise" / "-conservative" / "-no-denoise" variants,
commonly distributed as .pth on OpenModelDB tagged "RealCUGAN").

NOTE ON CONFIDENCE: RealCUGAN ships several structurally different
generators for different scales (UpCunet2x, UpCunet3x, UpCunet4x, and a
lightweight "fast" variant for 2x). This file implements UpCunet2x fully
(the most commonly distributed variant) and detects the others by
signature so the loader fails loudly with a clear message instead of
silently mis-building, rather than guessing at their internals.
"""

import torch
from torch import nn as nn
from torch.nn import functional as F

from .base import BaseSRArch
from .registry import ARCH2_REGISTRY


class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.seblock = SEBlock(out_channels, reduction=8, bias=True) if se else None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        return x * x0


class UNet1(nn.Module):
    """The generator body used by UpCunet2x."""

    def __init__(self, in_channels=3, out_channels=3, deconv=True):
        super().__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0) if not deconv else None
        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x1 = x1[:, :, 4:-4, 4:-4]
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z


class UpCunet2x(nn.Module):
    """RealCUGAN 2x generator. Input is padded externally by the caller
    (RealCUGAN requires specific padding/tiling handled by its own
    inference wrapper, not the standard RealESRGANer tile_process)."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)

    def forward(self, x):
        return self.unet1(x)


class RealCUGANArch(BaseSRArch):
    ARCH_NAME = 'realcugan'
    FAMILY = 'realcugan'

    @classmethod
    def detect(cls, state_dict):
        keys = state_dict.keys()
        # Signature unique to the UNet1/UpCunet2x block naming
        strong_sig = all(k in keys for k in (
            'unet1.conv1.conv.0.weight',
            'unet1.conv2.seblock.conv1.weight',
            'unet1.conv_bottom.weight',
        ))
        if strong_sig:
            return 1.0
        # UpCunet3x/4x use unet1 + unet2 - detect family but flag as unsupported build
        has_two_unets = any(k.startswith('unet1.') for k in keys) and any(k.startswith('unet2.') for k in keys)
        if has_two_unets:
            return 0.3  # recognized family, but build() below will raise a clear error
        return 0.0

    @classmethod
    def build(cls, state_dict, **overrides):
        keys = state_dict.keys()
        if any(k.startswith('unet2.') for k in keys):
            raise NotImplementedError(
                'This checkpoint looks like RealCUGAN 3x/4x (unet1 + unet2 cascade), '
                'which is not yet implemented in this compat layer - only the 2x '
                '(single UNet1) variant is supported. Please open an issue / add '
                'UpCunet3x/UpCunet4x to realcugan_arch.py before using this checkpoint.'
            )
        in_ch = state_dict['unet1.conv1.conv.0.weight'].shape[1]
        out_ch = state_dict['unet1.conv_bottom.weight'].shape[0] \
            if state_dict['unet1.conv_bottom.weight'].dim() == 4 \
            else state_dict['unet1.conv_bottom.weight'].shape[1]

        cfg = dict(in_channels=in_ch, out_channels=out_ch)
        cfg.update(overrides)

        model = UpCunet2x(**cfg)
        model.load_state_dict(state_dict, strict=True)
        return model


ARCH2_REGISTRY.register(RealCUGANArch)
