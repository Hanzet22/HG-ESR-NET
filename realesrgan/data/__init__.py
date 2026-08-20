import importlib
from os import path as osp


def _scandir_dataset_files(folder, suffix):
    import os
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.endswith(suffix):
            yield entry.name


# automatically scan and import dataset modules for registry (basicsr-free scandir)
dataset_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(v)[0] for v in _scandir_dataset_files(dataset_folder, '_dataset.py')]
# NOTE: the modules themselves (realesrgan_dataset.py / realesrgan_model.py) still
# import basicsr internally (DiffJPEG, USMSharp, SRGANModel, etc) - that's expected,
# they're training-time only. See realesrgan/__init__.py for how this is made optional.
_dataset_modules = [importlib.import_module(f'realesrgan.data.{file_name}') for file_name in dataset_filenames]
