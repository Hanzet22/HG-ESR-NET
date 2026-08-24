# Attributions & Credits

This project is a fork of **Real-ESRGAN**, modified to add universal
architecture support (auto-detecting and loading 25+ super-resolution
architectures from any `.pth` / `.safetensors` / `.ckpt` checkpoint).
None of the architecture detection/inference code in this fork was written
from scratch where a suitable open-source implementation already existed -
it stands entirely on the work of the projects and people below.

## Original project

**Real-ESRGAN**
Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan
https://github.com/xinntao/Real-ESRGAN
License: BSD-3-Clause
> Wang, Xintao, Liangbin Xie, Chao Dong, and Ying Shan. "Real-ESRGAN: Training
> real-world blind super-resolution with pure synthetic data." ICCVW 2021.

This fork keeps Real-ESRGAN's original training pipeline
(`realesrgan/models/`, `realesrgan/data/`) untouched and still credited to
the authors above. Only the inference/model-loading path was rebuilt.

## Universal architecture loading

**BasicSR**
Xintao Wang, Liangbin Xie, Ke Yu, Kelvin C.K. Chan, Chen Change Loy, Chao Dong
https://github.com/XPixelGroup/BasicSR
License: Apache-2.0
Reference implementations of RRDBNet, EDSR, RCAN, SRResNet, SwinIR, and the
`arch_util.py` helpers (`pixel_unshuffle`, residual blocks, etc.) that this
fork's `archs2/` fallback registry is based on.

**Spandrel**
chaiNNer team (Joey Ballentine / joeyballentine, RunDevelopment, and
contributors)
https://github.com/chaiNNer-org/spandrel
License: MIT
The primary architecture-detection and loading engine used by this fork
(`realesrgan/universal_loader.py`). Spandrel auto-detects architecture and
hyperparameters directly from a checkpoint's state_dict and is what makes
this project's "one loader for 25+ architectures" goal actually work -
extracted from and built on model support originally implemented in
**chaiNNer** (https://github.com/chaiNNer-org/chaiNNer).

**spandrel_extra_arches**
chaiNNer team
https://pypi.org/project/spandrel-extra-arches/
License: MIT (package) - bundles architecture code released under a mix of
restrictive / non-commercial licenses. Kept as a strictly optional,
separately-installed dependency in this fork for that reason - see the
"Licensing note" below before enabling it.

**OpenModelDB**
The OpenModelDB community / contributors
https://openmodeldb.info
Used in this fork only as an optional convenience utility (`openmodeldb`
PyPI package) for browsing/downloading/converting checkpoint formats - not
for architecture detection or inference, which is handled by Spandrel.

## Individual architectures (via Spandrel / spandrel_extra_arches)

Every architecture below is detected and run through Spandrel or
spandrel_extra_arches, not reimplemented in this fork. Credit goes to each
architecture's original authors; see the linked papers/repos for full
citations.

| Architecture | Authors / Paper |
|---|---|
| ESRGAN / RRDBNet | Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" (ECCVW 2018) |
| SRVGGNetCompact | Xintao Wang et al. (Real-ESRGAN project) |
| SPAN | Cheng Wan et al., "Swift Parameter-free Attention Network" |
| OmniSR | Hang Wang et al., "Omni Aggregation Networks for Lightweight Image Super-Resolution" |
| SAFMN | Long Sun et al., "Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution" |
| RCAN | Yulun Zhang et al., "Image Super-Resolution Using Very Deep Residual Channel Attention Networks" (ECCV 2018) |
| EDSR | Bee Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution" (CVPRW 2017) |
| SRResNet / SRGAN | Christian Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" (CVPR 2017) |
| HAT | Xiangyu Chen et al., "Activating More Pixels in Image Super-Resolution Transformer" |
| DAT | Zheng Chen et al., "Dual Aggregation Transformer for Image Super-Resolution" |
| SwinIR | Jingyun Liang et al., "SwinIR: Image Restoration Using Swin Transformer" |
| Swin2SR | Marcos V. Conde et al., "Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration" |
| SRFormer | Yupeng Zhou et al., "SRFormer: Permuted Self-Attention for Single Image Super-Resolution" |
| RGT | Zheng Chen et al., "Recursive Generalization Transformer for Image Super-Resolution" |
| DRCT | Yi-Hsin Chen et al., "DRCT: Saving Image Super-Resolution away from Information Bottleneck" |
| ATD | Leheng Zhang et al., "Transcending the Limit of Local Window: Advanced Super-Resolution Using Adaptive Token Dictionary" |
| Real-CUGAN | bilibili / ailab, https://github.com/bilibili/ailab |
| PLKSR / RealPLKSR | Dongheon Lee et al., "Partial Large Kernel CNNs for Efficient Super-Resolution" |
| RestoreFormer | Zhouxia Wang et al., "RestoreFormer: High-Quality Blind Face Restoration From Undegraded Key-Value Pairs" |
| RetinexFormer | Yuanhao Cai et al., "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement" |
| SCUNet | Kai Zhang et al., "Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis" |
| GRL | Yawei Li et al., "Efficient and Explicit Modelling of Image Hierarchies for Image Restoration" (CVPR 2023) |
| DITN | Bin Chen et al., "Dual Interactive Transformer Network for Efficient Image Super-Resolution" |
| SeemoRe | Eduard Zamfir et al., "See More Details: Efficient Image Super-Resolution by Experts Mining" - note: the seemoredetails reference implementation is released under CC BY-NC-SA 4.0 (academic/non-commercial); confirm the license of any specific checkpoint before commercial use |
| MoSR / MoESR | Mixture-of-experts-based super-resolution architectures available via Spandrel |
| DCTLSA | Zheng Wang et al., "DCT-based Local and Sparse Attention" family of efficient SR architectures |

Everything in this table is reachable automatically once Spandrel and (for
restrictively-licensed entries) spandrel_extra_arches are installed - none
of it required writing custom architecture code in this fork. Spandrel's
own README documents any per-architecture license notes in more detail than
this table does; check there for anything not covered above.

For the complete and current list of architectures Spandrel supports
(this fork does not hardcode which ones are available - it's whatever
version of spandrel/spandrel_extra_arches you have installed), see:
https://github.com/chaiNNer-org/spandrel#supported-architectures

## This fork's own additions

Written specifically for this fork, not derived from the projects above:
- `realesrgan/universal_loader.py` - the hybrid loader that tries Spandrel
  first, falls back to `archs2/`, and adds a separate ONNXRuntime path.
- `realesrgan/archs2/` - a small hand-written fallback registry (RRDBNet,
  SRVGGCompact, RealCUGAN 2x/3x/4x) used only when Spandrel cannot
  recognize a checkpoint. Layer definitions here follow the reference
  implementations in BasicSR / the official bilibili/ailab RealCUGAN repo
  (credited above) as closely as possible; the RealCUGAN network classes
  are transcribed from the official `upcunet_v3.py` source, keeping only
  the plain forward-pass path (not RealCUGAN's own custom tiling
  implementation - this fork tiles via `RealESRGANer`'s standard
  `tile_process()` instead).
- `realesrgan/model_url.py` - `--model_url` support: resolving a model link
  (including Google Drive share-link normalization) to a local cached file.
- `realesrgan/logging_setup.py` - centralized `--log_level` / `--verbose`
  logging configuration.
- The automatic OOM-retry logic in `realesrgan/utils.py`'s `tile_process()`.
- `test_universal_loader.py`, the `--model_path` / `--model_url` / `--arch` /
  `--list_archs` / `--log_level` / `--verbose` CLI additions in
  `inference_realesrgan.py`, and the requirements.txt / setup.py updates
  for Python 3.11/3.12/3.13 support.

See [VERSION.md](VERSION.md) for the version-by-version breakdown of when
each of these was added.

## Licensing note (please read before redistributing)

This fork mixes code and dependencies under **different licenses**:

- Real-ESRGAN (this fork's base): BSD-3-Clause
- BasicSR: Apache-2.0 (only needed for training, see `requirements.txt`'s
  `train` extra)
- Spandrel: MIT
- spandrel_extra_arches: MIT package, but **bundles architecture code under
  restrictive / non-commercial licenses**. Review
  https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel_extra_arches/README.md
  before using this in anything beyond a personal or non-commercial
  open-source project - some architectures inside it are not cleared for
  commercial use.

If you redistribute this fork, keep this file and check the license terms
of any architecture your users' checkpoints might trigger, particularly if
`spandrel_extra_arches` is installed.
