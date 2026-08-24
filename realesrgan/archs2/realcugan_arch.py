"""
RealCUGAN (Real Cascaded U-Net for anime upscaling), self-contained.

Covers checkpoints from the bilibili/ailab Real-CUGAN project:
https://github.com/bilibili/ailab/blob/main/Real-CUGAN/upcunet_v3.py
(up2x/up3x/up4x, "-denoise" / "-conservative" / "-no-denoise" / "-pro"
variants, commonly distributed as .pth on OpenModelDB tagged "RealCUGAN").

PROVENANCE: the network architecture classes below (SEBlock, UNetConv,
UNet1, UNet1x3, UNet2, UpCunet2x, UpCunet3x, UpCunet4x __init__ + plain
forward()) are transcribed directly from the official upcunet_v3.py source
to match weight shapes exactly. Only the plain forward() path is kept -
the original file also ships several alternate tiling implementations
(forward_a/b/c/d split-computation, forward_gap_sync, forward_fast_rough,
8-bit "cache_mode" quantization) built for RealCUGAN's own custom
tile-stitching logic. Those are NOT reproduced here: this compat layer
tiles via RealESRGANer's standard tile_process() instead, so only the
network definition needed for load_state_dict() + a single un-tiled
forward pass is needed. If you tile a very large image with a RealCUGAN
model through this loader, expect the seams RealCUGAN's own tiling was
specifically designed to avoid (see the "no-cutting-line" note in their
README) - for seamless large-image tiling, use RealCUGAN's own inference
scripts instead.

NOTE ON alpha/pro PARAMETERS: the official forward() takes extra
tile_mode, cache_mode, alpha, and pro arguments used for post-processing
(denoise strength blending, the "pro" model's different output value
range). This wrapper always runs at alpha=1.0 (standard fixed generator
output) since those are runtime inference knobs, not part of the
network's weights - a caller wanting RealCUGAN's alpha-blending denoise
control should use RealCUGAN's own inference wrapper instead of this
compat layer.
"""

import torch
from torch import nn as nn
from torch.nn import functional as F

from .base import BaseSRArch
from .registry import ARCH2_REGISTRY


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        if 'Half' in x.type():
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        return torch.mul(x, x0)


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


class UNet1(nn.Module):
    """Generator stage used standalone by UpCunet2x and as the first stage
    of UpCunet4x (with 64 intermediate channels feeding a PixelShuffle head)."""

    def __init__(self, in_channels, out_channels, deconv):
        super().__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        return self.conv_bottom(x3)


class UNet1x3(nn.Module):
    """Generator stage used by UpCunet3x - identical to UNet1 except the
    final deconv upsamples by 3x (kernel 5, stride 3) instead of 2x."""

    def __init__(self, in_channels, out_channels, deconv):
        super().__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 5, 3, 2)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        return self.conv_bottom(x3)


class UNet2(nn.Module):
    """Deeper second-stage generator (4 conv blocks) used as `unet2` by
    UpCunet2x, UpCunet3x, and UpCunet4x alike."""

    def __init__(self, in_channels, out_channels, deconv):
        super().__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)
        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

    def forward(self, x, alpha=1.0):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-16, -16, -16, -16))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
        x2 = F.pad(x2, (-4, -4, -4, -4))
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x4 = self.conv4(x2 + x3)
        x4 = x4 * alpha
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)
        return self.conv_bottom(x5)


class UpCunet2x(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x, alpha=1.0):
        n, c, h0, w0 = x.shape
        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        xp = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), 'reflect')
        xu = self.unet1.forward(xp)
        x0 = self.unet2.forward(xu, alpha)
        xu = F.pad(xu, (-20, -20, -20, -20))
        out = torch.add(x0, xu)
        if w0 != pw or h0 != ph:
            out = out[:, :, :h0 * 2, :w0 * 2]
        return out


class UpCunet3x(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.unet1 = UNet1x3(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x, alpha=1.0):
        n, c, h0, w0 = x.shape
        ph = ((h0 - 1) // 4 + 1) * 4
        pw = ((w0 - 1) // 4 + 1) * 4
        xp = F.pad(x, (14, 14 + pw - w0, 14, 14 + ph - h0), 'reflect')
        xu = self.unet1.forward(xp)
        x0 = self.unet2.forward(xu, alpha)
        xu = F.pad(xu, (-20, -20, -20, -20))
        out = torch.add(x0, xu)
        if w0 != pw or h0 != ph:
            out = out[:, :, :h0 * 3, :w0 * 3]
        return out


class UpCunet4x(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.unet1 = UNet1(in_channels, 64, deconv=True)
        self.unet2 = UNet2(64, 64, deconv=False)
        self.ps = nn.PixelShuffle(2)
        self.conv_final = nn.Conv2d(64, 12, 3, 1, padding=0, bias=True)

    def forward(self, x, alpha=1.0):
        n, c, h0, w0 = x.shape
        x00 = x
        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        xp = F.pad(x, (19, 19 + pw - w0, 19, 19 + ph - h0), 'reflect')
        xu = self.unet1.forward(xp)
        x0 = self.unet2.forward(xu, alpha)
        x1 = F.pad(xu, (-20, -20, -20, -20))
        out = torch.add(x0, x1)
        out = self.conv_final(out)
        out = F.pad(out, (-1, -1, -1, -1))
        out = self.ps(out)
        if w0 != pw or h0 != ph:
            out = out[:, :, :h0 * 4, :w0 * 4]
        out = out + F.interpolate(x00, scale_factor=4, mode='nearest')
        return out


class RealCUGANArch(BaseSRArch):
    ARCH_NAME = 'realcugan'
    FAMILY = 'realcugan'

    # UpCunet4x's unet1 output feeds conv_final - this key is unique to the
    # 4x variant, since 4x also has unet1.*/unet2.* like 2x/3x do.
    _V4_SIGNATURE_KEY = 'conv_final.weight'

    @classmethod
    def detect(cls, state_dict):
        keys = state_dict.keys()
        has_unet1 = any(k.startswith('unet1.') for k in keys)
        has_unet2 = any(k.startswith('unet2.') for k in keys)
        if not (has_unet1 and has_unet2):
            return 0.0
        if 'unet1.conv1.conv.0.weight' not in state_dict:
            return 0.0
        return 1.0

    @classmethod
    def _variant(cls, state_dict):
        """Return '2x', '3x', or '4x' based on structural signature."""
        if cls._V4_SIGNATURE_KEY in state_dict:
            return '4x'
        # 2x vs 3x differ in unet1.conv_bottom's deconv kernel size:
        # UNet1 (2x) uses kernel 4, UNet1x3 (3x) uses kernel 5.
        bottom_w = state_dict.get('unet1.conv_bottom.weight')
        if bottom_w is not None and bottom_w.dim() == 4:
            if bottom_w.shape[2] == 5:
                return '3x'
            return '2x'
        return '2x'  # fallback default

    @classmethod
    def build(cls, state_dict, **overrides):
        variant = overrides.pop('variant', None) or cls._variant(state_dict)
        in_ch = state_dict['unet1.conv1.conv.0.weight'].shape[1]

        if variant == '4x':
            cfg = dict(in_channels=in_ch, out_channels=overrides.pop('out_channels', 3))
            cfg.update(overrides)
            model = UpCunet4x(**cfg)
        else:
            bw = state_dict['unet1.conv_bottom.weight']
            out_ch = bw.shape[1] if bw.dim() == 4 else bw.shape[0]
            cfg = dict(in_channels=in_ch, out_channels=out_ch)
            cfg.update(overrides)
            model = UpCunet3x(**cfg) if variant == '3x' else UpCunet2x(**cfg)

        model.load_state_dict(state_dict, strict=True)
        return model


ARCH2_REGISTRY.register(RealCUGANArch)
