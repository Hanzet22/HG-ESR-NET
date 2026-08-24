<p align="center">
  <img src="assets/realesrgan_logo.png" height=120>
</p>

<div align="center">

# HG-ESR-NET

**通用架构超分辨率推理引擎**

[English](README.md) | [简体中文](README_CN.md) | [Bahasa Indonesia](README_ID.md)

[![License](https://img.shields.io/github/license/Hanzet22/HG-ESR-NET.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](#-依赖与安装)
[![Version](https://img.shields.io/badge/version-1.2.0-orange.svg)](VERSION.md)
[![Open issue](https://img.shields.io/github/issues/Hanzet22/HG-ESR-NET)](https://github.com/Hanzet22/HG-ESR-NET/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/Hanzet22/HG-ESR-NET)](https://github.com/Hanzet22/HG-ESR-NET/issues)

</div>

**HG-ESR-NET** 是一个通用超分辨率推理引擎，基于 Real-ESRGAN 成熟的放大流程构建，
并扩展了一个混合架构检测层，可从单个模型文件中自动加载 **25 种以上的
超分辨率/图像修复架构** —— 无需手写任何架构代码。

只需指向来自 OpenModelDB、HuggingFace 或你自己训练的任意 `.pth` /
`.safetensors` / `.ckpt` 模型文件，它就会自动检测架构、构建网络并运行推理 ——
无论是 ESRGAN、SPAN、OmniSR、SwinIR、HAT、DAT、RealCUGAN，还是其支持的其他架构，
使用方式都完全相同。

---

## ✨ HG-ESR-NET 新增了什么

- **通用模型加载器**（`realesrgan/universal_loader.py`）—— 混合检测机制：
  优先尝试 [Spandrel](https://github.com/chaiNNer-org/spandrel)（25 种以上架构，
  直接从原始 state_dict 自动推断超参数），如果 Spandrel 无法识别，
  则回退到手写的 `archs2/` 注册表（RRDBNet、SRVGGCompact、RealCUGAN 2x/3x/4x）。
- **`--model_path` + `--arch`** —— 直接加载*任意*模型文件，并可选提供架构提示以处理边缘情况。
  不再需要每次尝试 OpenModelDB 上的新模型时手动修改推理脚本添加新的 `nn.Module`。
- **`--model_url`** —— 直接通过链接下载模型，而不必先手动下载到本地。
  Google Drive 分享链接（OpenModelDB/社区模型最常见的分发方式）会自动转换为
  直接下载链接，并缓存在 `weights/from_url/` 目录下以便复用。
- **`.pth` / `.safetensors` / `.onnx` / `.ckpt`** 开箱即用。ONNX 模型通过独立的
  ONNXRuntime 会话运行（见 `load_onnx_model()`），因为 ONNX 计算图并非 PyTorch
  的 state_dict，理应拥有独立、诚实的代码路径，而不是被强行塞进同一个检测器中。
- **自动 OOM（显存不足）恢复** —— 处理过程中某个图块（tile）显存不足时，
  会自动以更小的子图块尺寸重试，而不是导致输出损坏或整个任务崩溃。
  批量处理时也会在图片之间主动释放显存，减少显存碎片。
- **`--log_level` / `--verbose`** —— 结构化、带时间戳的日志，
  可从安静模式（仅显示错误）调整到完全详细模式（显示每一次架构检测评分、
  缓存命中和重试记录）。
- **支持 Python 3.11 / 3.12 / 3.13** —— `requirements.txt` 和 `setup.py` 均已更新；
  `basicsr` 不再是推理所需的强制依赖（仅原始训练流程仍需要它，
  现作为可选的 `pip install .[train]` 附加项 —— 详见 [训练文档](docs/Training.md)）。
- **`test_universal_loader.py`** —— 环境自检脚本：验证你的环境
  （torch / spandrel / CUDA）、回退注册表是否正常，并可在正式使用前
  加载真实模型并进行一次试算（dummy forward pass）以验证其可用性。
- **fp16/fp32 切换**、分块处理（tiling）、预填充（pre-padding）等
  Real-ESRGAN 原有的推理选项均完整保留。

完整版本历史请见 **[VERSION.md](VERSION.md)**。

---

## 🔧 依赖与安装

- Python 3.11、3.12 或 3.13
- [PyTorch >= 2.2](https://pytorch.org/)（建议使用支持 CUDA 的版本以进行 GPU 推理）

### 安装

```bash
git clone https://github.com/Hanzet22/HG-ESR-NET.git
cd HG-ESR-NET
pip install -r requirements.txt
```

训练相关的额外依赖（仅微调训练时需要，推理不需要）：

```bash
pip install .[train]   # 会安装 basicsr、facexlib、gfpgan
```

### 首次使用前的自检

```bash
python test_universal_loader.py
# 通过后，再指向一个真实的模型文件进行测试：
python test_universal_loader.py --model_path weights/your_model.pth
```

---

## ⚡ 快速推理

### 使用任意模型（OpenModelDB / HuggingFace / 自己训练的模型）

```bash
python inference_realesrgan.py -i inputs/ --model_path weights/your_model.pth -o results/
```

架构和放大倍数会自动检测。如果检测结果不明确，可手动指定：

```bash
python inference_realesrgan.py -i inputs/ --model_path weights/model.pth --arch rrdbnet --scale 4
```

查看回退注册表已知的架构列表：

```bash
python inference_realesrgan.py --list_archs
```

### 通过链接使用模型（自动下载）

```bash
python inference_realesrgan.py -i inputs/ --model_url "https://drive.google.com/file/d/XXXXX/view?usp=sharing" -o results/
```

Google Drive 分享链接会自动转换为直接下载链接。下载后的文件会缓存在
`weights/from_url/` 目录下，因此使用相同的 `--model_url` 再次运行时会直接复用缓存，
无需重复下载。

### 使用官方 Real-ESRGAN 模型（自动下载）

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```

```console
用法：python inference_realesrgan.py -n 模型名称 -i 输入路径 -o 输出路径 [选项]...
  或：python inference_realesrgan.py --model_path 路径 -i 输入路径 -o 输出路径 [选项]...
  或：python inference_realesrgan.py --model_url 链接 -i 输入路径 -o 输出路径 [选项]...

  -h                   显示帮助信息
  -i --input           输入图片或文件夹。默认：inputs
  -o --output          输出文件夹。默认：results
  -n --model_name      官方模型名称（自动下载）。若设置了 --model_path/--model_url 则忽略此项。
  --model_path         任意本地 .pth/.ckpt/.pt/.safetensors 模型文件路径。
  --model_url           通过链接下载模型（自动转换 Google Drive 分享链接）。优先级高于 --model_path/-n。
  --arch               为 archs2/ 回退注册表提供的可选架构提示。
  --scale              放大倍数，通常从模型文件中自动检测。
  --list_archs         列出 archs2/ 回退注册表中的架构名称并退出。
  -s, --outscale       最终图像放大倍数。默认：4
  --suffix             输出图片文件名后缀。默认：out
  -t, --tile           分块（tile）尺寸，0 表示不分块。默认：0
  --face_enhance       是否使用 GFPGAN 进行人脸增强。默认：False
  --fp32               使用 fp32 精度进行推理。默认：fp16（半精度）。
  --ext                输出图片扩展名。可选：auto | jpg | png。默认：auto
  --log_level          日志详细程度：debug | info | warning | error。默认：info。
  --verbose            等同于 --log_level debug。
```

结果将保存至 `results/` 文件夹。当某个图块处理时显存不足，系统会自动以
更小的尺寸重试，而不会导致整张图片处理失败 —— 详情请见 [VERSION.md](VERSION.md)。

---

## 🏰 支持的架构

通过 Spandrel 自动处理（无需任何配置）：ESRGAN/RRDBNet、SRVGGCompact、SPAN、
OmniSR、SAFMN、HAT、DAT、SwinIR、Swin2SR、SRFormer、RGT、DRCT、ATD、
RealCUGAN（2x/2x_fast/3x/4x）、PLKSR/RealPLKSR、RestoreFormer、RetinexFormer、
SCUNet、GRL、DITN、SeemoRe、MoSR、MoESR、DCTLSA 等等 —— 完整最新列表请见
[Spandrel 支持的架构](https://github.com/chaiNNer-org/spandrel#supported-architectures)。

手写回退注册表（`archs2/`，仅在 Spandrel 无法识别模型时使用）：RRDBNet、
SRVGGCompact、RealCUGAN 2x/3x/4x。

受限制/非商业许可证的架构可通过独立的 `spandrel_extra_arches` 包获取 ——
在将其用于个人/非商业用途之外的场景前，请务必阅读
[ATTRIBUTIONS.md](ATTRIBUTIONS.md) 中的许可说明。

---

## 📧 联系方式

关于 **HG-ESR-NET** 的问题或反馈：`farhanzet4@gmail.com` 或
`hypergarudatkj@gmail.com`。

关于本分支所基于的原始 Real-ESRGAN 项目的问题，请见下方致谢部分。

---

## 🙏 基于 / 致谢

HG-ESR-NET 是 **Real-ESRGAN** 的一个分支（fork），其推理/模型加载部分
围绕通用架构检测器进行了重构。原始训练流程、论文及上游项目均在此完整致谢。
每个架构与所用依赖库的完整署名信息见 [ATTRIBUTIONS.md](ATTRIBUTIONS.md)——
在二次分发本项目前请务必阅读该文件。

### Real-ESRGAN（原始项目）

> Wang, Xintao, Liangbin Xie, Chao Dong, and Ying Shan. "Real-ESRGAN:
> Training real-world blind super-resolution with pure synthetic data."
> ICCVW 2021.

[[论文](https://arxiv.org/abs/2107.10833)] &emsp; [[原始仓库](https://github.com/xinntao/Real-ESRGAN)]

[Xintao Wang](https://xinntao.github.io/)、Liangbin Xie、
[Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ)、
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) ——
腾讯 ARC Lab；中国科学院深圳先进技术研究院

```bibtex
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```

原始项目联系方式：`xintao.wang@outlook.com` 或 `xintaowang@tencent.com`。

### 本项目所依赖的核心组件

- **[BasicSR](https://github.com/XPixelGroup/BasicSR)**（XPixelGroup /
  Xintao Wang、Liangbin Xie、Ke Yu、Kelvin C.K. Chan、Chen Change Loy、
  Chao Dong）—— `archs2/` 回退实现所参考的官方架构实现。
- **[Spandrel](https://github.com/chaiNNer-org/spandrel)**（chaiNNer 团队）——
  为本项目通用模型支持提供核心动力的架构自动检测与加载引擎。
- **[OpenModelDB](https://openmodeldb.info)** —— 用于浏览/下载/转换模型文件的
  可选辅助工具。

完整的架构、作者、论文及许可条款列表请见 **[ATTRIBUTIONS.md](ATTRIBUTIONS.md)**。

### 推荐的相关项目

- [GFPGAN](https://github.com/TencentARC/GFPGAN) —— 实用的人脸复原算法
- [BasicSR](https://github.com/XPixelGroup/BasicSR) —— 开源的图像/视频修复工具箱
- [Spandrel](https://github.com/chaiNNer-org/spandrel) —— 通用 PyTorch 超分辨率/图像修复模型加载库
- [OpenModelDB](https://openmodeldb.info) —— 社区模型数据库
