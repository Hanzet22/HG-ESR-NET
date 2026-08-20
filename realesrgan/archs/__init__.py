import importlib
import warnings
from os import path as osp

# automatically scan and import arch modules for registry (basicsr-free scandir)
# scan all the files that end with '_arch.py' under the archs folder
arch_folder = osp.dirname(osp.abspath(__file__))


def _scandir_arch_files(folder):
    import os
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.endswith('_arch.py'):
            yield entry.name


arch_filenames = [osp.splitext(v)[0] for v in _scandir_arch_files(arch_folder)]
# import all the arch modules. Some of these (discriminator_arch.py, srvgg_arch.py)
# are training-time-only and register themselves against basicsr's ARCH_REGISTRY,
# so they hard-depend on basicsr being installed. That's fine for training setups,
# but inference-only users (see requirements.txt - basicsr is now an optional
# `train` extra, not a core dependency) shouldn't have `import realesrgan` crash
# over it. Skip + warn per-module instead of failing the whole package import.
_arch_modules = []
for _file_name in arch_filenames:
    try:
        _arch_modules.append(importlib.import_module(f'realesrgan.archs.{_file_name}'))
    except ImportError as _e:
        warnings.warn(
            f'realesrgan.archs.{_file_name} could not be imported ({_e}) - likely because '
            f'it is a training-time module that needs basicsr (pip install .[train]). '
            f'Skipping; this does not affect inference via realesrgan.universal_loader.',
            stacklevel=2,
        )
