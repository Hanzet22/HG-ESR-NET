"""
SRVGGNetCompact (the "Compact" family), self-contained.

Covers checkpoints such as:
  - realesr-general-x4v3, realesr-animevideov3
  - Most "-compact" tagged uploads on OpenModelDB
"""

import re
from torch import nn as nn
from torch.nn import functional as F

from .base import BaseSRArch
from .registry import ARCH2_REGISTRY


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network for super-resolution (upsample in the last layer)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._make_act(act_type, num_feat))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._make_act(act_type, num_feat))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _make_act(act_type, num_feat):
        if act_type == 'relu':
            return nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            return nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError(f'Unsupported act_type: {act_type}')

    def forward(self, x):
        out = x
        for i in range(len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


class SRVGGNetCompactArch(BaseSRArch):
    ARCH_NAME = 'srvgg_compact'
    FAMILY = 'esrgan'

    @classmethod
    def detect(cls, state_dict):
        keys = list(state_dict.keys())
        if not all(k.startswith('body.') or k == 'body' for k in keys if '.' in k or k == 'body'):
            # allow only body.* keys (plus possibly nothing else) - strong signature
            non_body = [k for k in keys if not k.startswith('body.')]
            if non_body:
                return 0.0
        has_body_keys = any(re.match(r'^body\.\d+\.weight$', k) for k in keys)
        has_upsampler = not any('upsampler' in k for k in keys)  # PixelShuffle has no params
        return 0.9 if has_body_keys and has_upsampler else 0.0

    @classmethod
    def build(cls, state_dict, **overrides):
        # locate every body.N.weight that is a conv (4D tensor) to figure out feat/conv count
        conv_indices = []
        for k, v in state_dict.items():
            m = re.match(r'^body\.(\d+)\.weight$', k)
            if m and v.dim() == 4:
                conv_indices.append((int(m.group(1)), v))
        conv_indices.sort(key=lambda x: x[0])

        if not conv_indices:
            raise ValueError('srvgg_compact: no conv weights found in state_dict')

        first_idx, first_w = conv_indices[0]
        last_idx, last_w = conv_indices[-1]
        num_in_ch = first_w.shape[1]
        num_feat = first_w.shape[0]

        out_and_scale = last_w.shape[0]  # num_out_ch * upscale^2
        # infer act type by presence of PReLU weight (1D, matches num_feat) between conv layers
        has_prelu = any(v.dim() == 1 and v.shape[0] == num_feat for k, v in state_dict.items()
                         if re.match(r'^body\.\d+\.weight$', k))
        act_type = overrides.pop('act_type', 'prelu' if has_prelu else 'leakyrelu')

        # num_conv = number of interior conv layers (exclude first + last conv indices' layer count)
        # Each conv layer occupies one body index; activations occupy the next index but have
        # no '.weight' matching a 4D tensor, so conv_indices directly gives us layer positions.
        num_conv = len(conv_indices) - 2  # minus first and last conv

        num_out_ch = overrides.pop('num_out_ch', 3)
        upscale = overrides.pop('upscale', None)
        if upscale is None:
            # out_and_scale = num_out_ch * upscale^2 -> try common scales
            for candidate in (1, 2, 3, 4, 8):
                if out_and_scale == num_out_ch * candidate * candidate:
                    upscale = candidate
                    break
            if upscale is None:
                upscale = 4  # fallback default

        cfg = dict(num_in_ch=num_in_ch, num_out_ch=num_out_ch, num_feat=num_feat,
                   num_conv=num_conv, upscale=upscale, act_type=act_type)
        cfg.update(overrides)

        model = SRVGGNetCompact(**cfg)
        model.load_state_dict(state_dict, strict=True)
        return model


ARCH2_REGISTRY.register(SRVGGNetCompactArch)
