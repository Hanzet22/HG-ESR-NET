"""
Centralized logging setup for HG-ESR-NET.

Every module in this project gets its logger via `logging.getLogger('realesrgan.<name>')`
(see universal_loader.py, utils.py, model_url.py, archs2/registry.py for examples) -
this module is only responsible for configuring HOW those loggers render:
format, level, and where they write to. Call `setup_logging()` once, early,
from the CLI entry point (inference_realesrgan.py's main()); library code
that imports realesrgan directly (not via the CLI) can call it too, or just
rely on Python's default logging behavior (WARNING+ to stderr) if it never
calls setup_logging() at all - this module never configures logging as an
import side-effect, only when explicitly asked to.
"""

import logging
import sys

# Human-readable level names for --log_level, mapped to logging module constants.
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
}

_configured = False


def setup_logging(level: str = 'info', verbose: bool = False):
    """Configure logging for the whole `realesrgan` logger hierarchy.

    Parameters
    ----------
    level : str
        One of 'debug', 'info', 'warning', 'error'. Ignored if verbose=True
        (verbose forces 'debug').
    verbose : bool
        Shortcut for --verbose on the CLI: forces DEBUG level regardless of
        `level`, so every internal decision (arch detection scores, OOM
        retry attempts, cache hits, etc.) is printed.

    Safe to call more than once - only the first call actually attaches a
    handler; subsequent calls just adjust the level, so re-running this in
    a notebook cell doesn't produce duplicate log lines.
    """
    global _configured

    effective_level = logging.DEBUG if verbose else LOG_LEVELS.get(level.lower(), logging.INFO)

    root = logging.getLogger('realesrgan')
    root.setLevel(effective_level)

    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S',
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.propagate = False  # don't also send through the root Python logger (avoid dupes)
        _configured = True
    else:
        for handler in root.handlers:
            handler.setLevel(effective_level)

    return root
