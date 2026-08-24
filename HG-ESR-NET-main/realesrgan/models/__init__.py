import importlib
from os import path as osp


def _scandir_model_files(folder, suffix):
    import os
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.endswith(suffix):
            yield entry.name


# automatically scan and import model modules for registry (basicsr-free scandir)
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(v)[0] for v in _scandir_model_files(model_folder, '_model.py')]
# NOTE: the modules themselves (realesrgan_dataset.py / realesrgan_model.py) still
# import basicsr internally (DiffJPEG, USMSharp, SRGANModel, etc) - that's expected,
# they're training-time only. See realesrgan/__init__.py for how this is made optional.
_model_modules = [importlib.import_module(f'realesrgan.models.{file_name}') for file_name in model_filenames]
