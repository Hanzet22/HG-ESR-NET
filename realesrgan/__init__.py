# flake8: noqa
"""
realesrgan package init.

Inference-only usage (the universal compat layer: universal_loader, utils.RealESRGANer,
archs2/ registry) does NOT require basicsr and is imported unconditionally below.

Training-time modules (data/, models/) still depend on basicsr (DiffJPEG, USMSharp,
SRGANModel, degradation pipeline, etc - see requirements.txt's `train` extra:
pip install .[train]). They are imported lazily/optionally here so that a plain
`import realesrgan` for upscaling doesn't force-install basicsr just to fail on
an unrelated training-only import.
"""

from .archs import *
from .utils import *
from .version import *

try:
    from .data import *
    from .models import *
except ImportError as _e:
    import warnings
    warnings.warn(
        f'realesrgan.data / realesrgan.models (training-time, basicsr-dependent) '
        f'could not be imported: {_e}. This is fine for inference/upscaling. '
        f'Install the training extras with: pip install .[train]',
        stacklevel=2,
    )
