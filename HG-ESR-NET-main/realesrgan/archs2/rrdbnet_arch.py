"""
RRDBNet (ESRGAN / Real-ESRGAN family), self-contained re-implementation.

Covers checkpoints such as:
  - RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B, RealESRGAN_x2plus
  - Original ESRGAN / BSRGAN weights that share the RRDB block design
  - Most "x4plus"-style community fine-tunes on OpenModelDB

This does not import basicsr - hyperparameters (num_feat, num_block,
num_grow_ch, scale) are inferred directly from the state_dict shapes so
arbitrary community checkpoints of this family load without a config file.
"""

import re
import torch
from torch import nn as nn
from torch.nn import functional as F

from .base import BaseSRArch
from .registry import ARCH2_REGISTRY


def _make_layer(block, n_layers, **kwargs):
    return nn.Sequential(*[block(**kwargs) for _ in range(n_layers)])


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class RRDBNet(nn.Module):
    """Generator used by ESRGAN / Real-ESRGAN. scale in {1, 2, 4} (x1/x2 pixel-unshuffle
    the input before the body to keep the network operating at a fixed internal scale)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = _make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RRDBNetArch(BaseSRArch):
    ARCH_NAME = 'rrdbnet'
    FAMILY = 'esrgan'

    _RDB_KEY_RE = re.compile(r'^body\.\d+\.rdb1\.conv1\.weight$')

    @classmethod
    def detect(cls, state_dict):
        keys = state_dict.keys()
        # Strong signature: RRDB-specific key names unique to this family
        has_core = all(k in keys for k in ('conv_first.weight', 'conv_body.weight',
                                            'conv_up1.weight', 'conv_last.weight'))
        if not has_core:
            return 0.0
        has_rdb = any(cls._RDB_KEY_RE.match(k) for k in keys)
        return 1.0 if has_rdb else 0.4  # 0.4 = looks close but body block shape unconfirmed

    @classmethod
    def build(cls, state_dict, **overrides):
        num_feat = state_dict['conv_first.weight'].shape[0]
        num_out_ch = state_dict['conv_last.weight'].shape[0]

        # infer num_block by counting the highest "body.N." index present
        block_indices = set()
        for k in state_dict.keys():
            m = re.match(r'^body\.(\d+)\.', k)
            if m:
                block_indices.add(int(m.group(1)))
        num_block = (max(block_indices) + 1) if block_indices else 23

        num_grow_ch = state_dict['body.0.rdb1.conv1.weight'].shape[0] if 'body.0.rdb1.conv1.weight' in state_dict \
            else 32

        conv_first_in = state_dict['conv_first.weight'].shape[1]
        # conv_first_in = num_in_ch * (4 or 16) if pixel-unshuffled, else num_in_ch directly.
        # Standard checkpoints: 12 -> scale=2 (3*4), 48 -> scale=1 (3*16), else scale=4.
        if conv_first_in == 12:
            scale = 2
            num_in_ch = 3
        elif conv_first_in == 48:
            scale = 1
            num_in_ch = 3
        else:
            scale = 4
            num_in_ch = conv_first_in

        cfg = dict(num_in_ch=num_in_ch, num_out_ch=num_out_ch, scale=scale,
                   num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch)
        cfg.update(overrides)  # manual overrides win

        model = RRDBNet(**cfg)
        model.load_state_dict(state_dict, strict=True)
        return model


ARCH2_REGISTRY.register(RRDBNetArch)
