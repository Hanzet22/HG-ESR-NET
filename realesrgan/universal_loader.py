"""
Universal model loader for the Real-ESRGAN compat layer.

Load order (hybrid, mirrors the --arch hybrid mode in archs2/registry.py):

    1. Read the checkpoint file (.pth / .safetensors / .onnx / .ckpt) into
       a raw state_dict, regardless of extension.
    2. Try `spandrel` first - it auto-detects architecture + hyperparameters
       for 25+ community/official SR architectures (ESRGAN, SPAN, OmniSR,
       SAFMN, HAT, DAT, SwinIR, Swin2SR, SRFormer, RGT, DRCT, ATD,
       RealCUGAN x2/x2_fast/x3/x4, PLKSR/RealPLKSR, RestoreFormer,
       RetinexFormer, SCUNet, ...).
    3. If spandrel doesn't recognize the checkpoint (raises
       UnsupportedModelError) OR the caller passed --arch pointing at
       something only our own registry knows about, fall back to the
       hand-written archs2/ registry (RRDBNet, SRVGGCompact, RealCUGAN 2x -
       kept as a safety net, not the primary path anymore).
    4. If neither recognizes it, raise a clear combined error.

ONNX note: spandrel and archs2/ both work on PyTorch state_dicts. .onnx
files are graph+weights, not a state_dict, so they're routed to a separate
ONNXRuntime inference wrapper instead of being shoved through the same
detector - trying to detect a PyTorch arch signature from an ONNX graph
would be unreliable and dishonest.
"""

import logging
import os

import torch

# Module-level guard so spandrel_extra_arches.install() (which errors if
# called twice) only ever runs once per process, across every load_sr_model() call.
_EXTRA_ARCHES_INSTALLED = False

logger = logging.getLogger('realesrgan.universal_loader')


class UniversalLoadError(Exception):
    """Raised when no backend (spandrel, archs2 fallback) can load the checkpoint."""
    pass


# ---------------------------------------------------------------------------
# Step 1: raw file -> state_dict, for every supported extension
# ---------------------------------------------------------------------------

def _read_checkpoint(model_path: str, device='cpu'):
    """Read a .pth/.ckpt/.pt/.safetensors file into a raw state_dict (dict[str, Tensor]).
    Does NOT normalize wrapping (params_ema/params/etc) - callers normalize
    via BaseSRArch.strip_prefix() or let spandrel handle its own unwrapping.
    """
    ext = os.path.splitext(model_path)[1].lower()

    if ext == '.safetensors':
        from safetensors.torch import load_file
        return load_file(model_path, device=device)

    if ext in ('.pth', '.ckpt', '.pt'):
        # weights_only=True where possible: community checkpoints are
        # frequently untrusted pickle files, and full unpickling can execute
        # arbitrary code. Fall back to weights_only=False only if that fails
        # (some older checkpoints pickle non-tensor metadata alongside
        # weights), and warn loudly when doing so.
        try:
            return torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e:
            logger.warning(
                f'[universal_loader] weights_only=True load failed for {model_path} '
                f'({e}); retrying with weights_only=False. Only do this for '
                f'checkpoints from a source you trust - full unpickling can '
                f'execute arbitrary code.'
            )
            return torch.load(model_path, map_location=device, weights_only=False)

    raise UniversalLoadError(
        f'Unsupported checkpoint extension "{ext}" for state_dict loading. '
        f'Supported: .pth, .ckpt, .pt, .safetensors. Use load_onnx_model() '
        f'for .onnx files instead.'
    )


# ---------------------------------------------------------------------------
# Step 2/3: hybrid load - spandrel primary, archs2/ fallback
# ---------------------------------------------------------------------------

def load_sr_model(model_path: str, arch_hint: str = None, device='cpu', **archs2_overrides):
    """Load any supported PyTorch super-resolution checkpoint.

    Parameters
    ----------
    model_path : str
        Path to a .pth / .ckpt / .pt / .safetensors file.
    arch_hint : str or None
        Optional manual architecture name.
        - If it matches a spandrel-known arch id, currently unused for
          forcing spandrel (spandrel's own detection is already
          near-exhaustive and doesn't expose a "force this arch" hook in
          its public API) - spandrel is always tried first regardless.
        - If spandrel fails to recognize the file, arch_hint is passed to
          the archs2/ fallback registry's hybrid loader (see
          ArchRegistry.load in archs2/registry.py).
    device : str
        Device to map tensors to on load ('cpu', 'cuda', 'cuda:0', ...).
    archs2_overrides :
        Extra kwargs forwarded to the archs2/ fallback's build() if that
        path is taken (e.g. num_block=23 to override a mis-detected count).

    Returns
    -------
    (model: nn.Module, meta: dict)
        meta always contains: {'backend': 'spandrel' | 'archs2', 'arch': str,
        'scale': int | None, 'confidence': float}
        scale is None for archs2-loaded models where the arch class doesn't
        expose it directly on the descriptor (caller should read model.scale
        if needed, RRDBNet/SRVGGCompact both set self.upscale or self.scale).
    """
    # --- Try spandrel first ---
    try:
        from spandrel import ImageModelDescriptor, ModelLoader
        # Registering spandrel_extra_arches must happen exactly once per
        # process (calling .install() twice raises), so guard with a module
        # global instead of relying on import machinery alone.
        global _EXTRA_ARCHES_INSTALLED
        if not _EXTRA_ARCHES_INSTALLED:
            try:
                import spandrel_extra_arches
                spandrel_extra_arches.install()
                _EXTRA_ARCHES_INSTALLED = True
                logger.info('[universal_loader] spandrel_extra_arches installed - '
                             'restrictive/non-commercial-licensed architectures now available.')
            except ImportError:
                _EXTRA_ARCHES_INSTALLED = True  # nothing to retry - avoid re-checking every call
                logger.info(
                    '[universal_loader] spandrel_extra_arches not installed - '
                    'architectures under non-commercial/restrictive licenses '
                    'will not be available. pip install spandrel_extra_arches '
                    'to enable them (review their licenses first).'
                )

        descriptor = ModelLoader(device=device).load_from_file(model_path)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise UniversalLoadError(
                f'{model_path} loaded via spandrel but is not an image-to-image '
                f'model (got {type(descriptor).__name__}) - not usable for '
                f'upscaling in this pipeline.'
            )
        model = descriptor.model
        meta = {
            'backend': 'spandrel',
            'arch': descriptor.architecture.name if hasattr(descriptor.architecture, 'name')
                    else str(descriptor.architecture),
            'scale': getattr(descriptor, 'scale', None),
            'confidence': 1.0,  # spandrel's detection is exact-match based, not scored
        }
        logger.info(f'[universal_loader] Loaded via spandrel: arch={meta["arch"]}, scale={meta["scale"]}')
        return model, meta

    except ImportError:
        logger.warning(
            '[universal_loader] spandrel is not installed - falling back to the '
            'built-in archs2/ registry only (RRDBNet, SRVGGCompact, RealCUGAN 2x). '
            'pip install spandrel to unlock 25+ architectures.'
        )
    except Exception as e:
        # covers spandrel.UnsupportedModelError and any other load-time failure
        logger.info(f'[universal_loader] spandrel could not load {model_path} ({e}); trying archs2/ fallback.')

    # --- Fallback: archs2/ hand-written registry ---
    from .archs2.registry import ARCH2_REGISTRY, AmbiguousArchError, UnknownArchError
    # ensure fallback archs are registered (import triggers their
    # ARCH2_REGISTRY.register(...) calls at module load time)
    from .archs2 import rrdbnet_arch, srvgg_arch, realcugan_arch  # noqa: F401

    state_dict = _read_checkpoint(model_path, device=device)
    try:
        model, arch_name, confidence = ARCH2_REGISTRY.load(state_dict, arch_hint=arch_hint, **archs2_overrides)
    except (AmbiguousArchError, UnknownArchError) as e:
        raise UniversalLoadError(
            f'Could not load {model_path}: neither spandrel nor the archs2/ '
            f'fallback recognized this checkpoint. archs2/ error: {e}'
        ) from e

    model.to(device)
    meta = {'backend': 'archs2', 'arch': arch_name, 'scale': getattr(model, 'scale', None) or getattr(model, 'upscale', None),
            'confidence': confidence}
    logger.info(f'[universal_loader] Loaded via archs2/ fallback: arch={arch_name} (confidence={confidence:.2f})')
    return model, meta


# ---------------------------------------------------------------------------
# ONNX path - separate, honest about being a different runtime
# ---------------------------------------------------------------------------

def load_onnx_model(model_path: str, providers=None):
    """Load a .onnx super-resolution model as an ONNXRuntime InferenceSession.

    Returns (session, meta). This is NOT a torch.nn.Module - the caller
    (inference_realesrgan.py) needs a separate code path for ONNX inference
    (session.run(...) instead of model(tensor)), since mixing this behind
    the same call signature as load_sr_model() would hide a real
    behavioral difference (no autograd, no .to(device), numpy in/out).
    """
    import onnxruntime as ort

    if providers is None:
        available = ort.get_available_providers()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available \
            else ['CPUExecutionProvider']

    session = ort.InferenceSession(model_path, providers=providers)
    input_meta = session.get_inputs()[0]
    meta = {
        'backend': 'onnxruntime',
        'input_name': input_meta.name,
        'input_shape': input_meta.shape,
        'providers': session.get_providers(),
    }
    return session, meta
