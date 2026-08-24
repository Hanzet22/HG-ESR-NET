# Version History

This file tracks what changed between versions of **HG-ESR-NET**, so future
updates (V1.3, V2, etc) have a clear record of what came before and why.

Format: newest first. Each entry lists what changed, not implementation
detail - see git commit history / `ATTRIBUTIONS.md` for the technical and
attribution specifics.

---

## V1.2

**Added**
- RealCUGAN 3x and 4x support in the `archs2/` fallback registry (previously
  only 2x was implemented; 3x/4x now transcribed from the official
  bilibili/ailab source).
- `--model_url` - download a model directly from a link instead of requiring
  a local file first. Automatically converts Google Drive share links to
  direct-download links. Downloads are cached under `weights/from_url/`.
- `--log_level` and `--verbose` - structured logging with configurable
  verbosity (debug/info/warning/error), replacing scattered bare `print()`
  calls with consistent, timestamped, filterable log output.
- Automatic OOM recovery during tile processing: a tile that runs out of
  VRAM is now automatically retried at a reduced sub-tile size (split into
  quadrants, recursively) instead of the run silently continuing with
  corrupted output or crashing on an unrelated error.
- `torch.cuda.empty_cache()` called automatically between images in a batch
  run, reducing memory fragmentation across a long run.
- Official Python 3.13 support (`setup.py` classifiers and
  `python_requires` updated; verified against real Colab runs on 3.13).
- Run summary at the end of a batch (`N succeeded, N skipped, N failed`).

**Fixed**
- `spandrel_extra_arches` was never actually being activated - the
  integration called a non-existent API (`EXTRA_REGISTRY` /
  `MAIN_REGISTRY.add`) instead of the real one (`spandrel_extra_arches.install()`).
  Architectures under restrictive/non-commercial licenses are now properly
  available when that package is installed.
- `tile_process()`'s OOM handling previously just `print()`-ed the error and
  continued execution with no `output_tile` assigned - a silent failure mode
  that could crash on an unrelated `UnboundLocalError` or corrupt output with
  a stale tile from the previous iteration. Now raises a clear, actionable
  error after automatic retry is exhausted.

---

## V1.1

**Added**
- Confirmed and documented Spandrel's much broader architecture coverage
  than initially scoped (GRL, DITN, SeemoRe, MoSR, MoESR, DCTLSA, RCAN, and
  more all covered automatically, alongside categories like face
  restoration, inpainting, denoising, dejpeg, colorization, dehazing, and
  low-light enhancement).

**Fixed**
- `inference_realesrgan.py` crashed with `AttributeError: 'NoneType' object
  has no attribute 'shape'` when a non-image file (e.g. a video) was present
  in the input folder, since `cv2.imread()` silently returns `None` instead
  of raising. Now skipped with a clear warning instead of crashing the
  whole batch.
- Removed stray `.pyc` / `__pycache__` files that had been accidentally
  bundled into release zips; confirmed `.gitignore` covers them and fixed
  it to stop ignoring `version.py` (which this project intentionally commits
  so `import realesrgan` works without a build step first).

---

## V1.0

**Initial release** - forked from
[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and rebuilt
around universal architecture support.

**Added**
- `realesrgan/universal_loader.py` - hybrid model loader: tries
  [Spandrel](https://github.com/chaiNNer-org/spandrel) first (25+
  architectures with automatic hyperparameter inference from the raw
  state_dict), falls back to a hand-written `archs2/` registry (RRDBNet,
  SRVGGCompact, RealCUGAN 2x) for anything Spandrel doesn't recognize, and a
  separate ONNXRuntime path for `.onnx` checkpoints.
- `--model_path` and `--arch` CLI flags - load any local checkpoint
  (`.pth` / `.ckpt` / `.pt` / `.safetensors`) with architecture
  auto-detected, with an optional manual override.
- `--list_archs` - list architectures known to the `archs2/` fallback.
- `test_universal_loader.py` - environment + model-loading sanity check
  script (import check, registry check, CUDA check, optional real-model
  load-and-forward-pass test).
- Python 3.11/3.12 support - `requirements.txt` and `setup.py` updated;
  `basicsr` made an optional `train` extra instead of a hard dependency
  (it's only needed for the original training pipeline, not inference).
- `ATTRIBUTIONS.md`, updated `CODE_OF_CONDUCT.md` and `LICENSE` (BSD-3-Clause,
  dual copyright) for the HG-ESR-NET fork.

**Verified**
- End-to-end tested on Google Colab (Tesla T4) across three architecture
  tiers: SRVGGCompact (lightweight), RRDBNet (medium), and HAT-L (heavy
  transformer-based) - all loading and running successfully via Spandrel
  with no manual architecture code required.
