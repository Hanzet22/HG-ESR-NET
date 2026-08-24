"""
URL-based model input for HG-ESR-NET.

Lets a caller point --model_url at a link a user was given (most commonly a
Google Drive share link, since that's how the vast majority of OpenModelDB
and community-trained checkpoints are distributed) instead of requiring
them to manually download the file and pass a local --model_path first.

This module only resolves a URL down to a local file path - it does NOT
touch architecture detection or loading (that's universal_loader.py's job).
"""

import hashlib
import logging
import os
import re

logger = logging.getLogger('realesrgan.model_url')

# Where downloaded-by-URL models are cached between runs, relative to the
# project root - separate from weights/ (used for the 6 official -n models)
# so the two caches don't collide or get confused with each other.
_URL_CACHE_SUBDIR = 'weights/from_url'


class ModelURLError(Exception):
    """Raised when a --model_url can't be resolved to a downloadable file."""
    pass


def _gdrive_file_id(url: str):
    """Extract the file ID from any common Google Drive share-link shape.
    Returns None if `url` doesn't look like a Google Drive link at all."""
    patterns = [
        r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',      # .../file/d/<id>/view
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',    # .../open?id=<id>
        r'drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)',      # already a direct-ish link
        r'[?&]id=([a-zA-Z0-9_-]+)',                          # any other ...?id=<id> variant
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def normalize_model_url(url: str) -> str:
    """Turn a user-facing share link into a direct-downloadable URL.

    Currently handles:
      - Google Drive share links (the overwhelming majority of OpenModelDB /
        community checkpoint links) -> https://drive.google.com/uc?export=download&id=<id>
      - Anything else is returned unchanged (assumed to already be a direct
        download link - GitHub releases, HuggingFace resolve/ URLs, etc.)
    """
    if 'drive.google.com' in url:
        file_id = _gdrive_file_id(url)
        if file_id is None:
            raise ModelURLError(
                f'This looks like a Google Drive link but no file ID could be extracted: {url}\n'
                'Expected a link shaped like https://drive.google.com/file/d/<ID>/view?usp=sharing '
                'or https://drive.google.com/open?id=<ID>.'
            )
        direct = f'https://drive.google.com/uc?export=download&id={file_id}'
        logger.info(f'[model_url] Normalized Google Drive link -> {direct}')
        return direct

    if not url.startswith('https://') and not url.startswith('http://'):
        raise ModelURLError(f'--model_url must be an http(s) URL, got: {url}')

    return url


def _cache_filename(url: str, file_id: str = None) -> str:
    """Deterministic local filename for a given URL, so re-running with the
    same --model_url reuses the cached download instead of re-fetching."""
    if file_id:
        base = f'gdrive_{file_id}'
    else:
        base = hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]
    return base  # extension is unknown until headers/content are seen; appended by the downloader


def resolve_model_url(url: str, root_dir: str, cache: bool = True) -> str:
    """Download (or reuse a cached download of) a model from `url`,
    returning a local file path suitable for RealESRGANer(model_path=...).

    Parameters
    ----------
    url : str
        Any share link or direct download URL for a model checkpoint.
    root_dir : str
        Project root - the cache lives under `<root_dir>/weights/from_url/`.
    cache : bool
        If True (default) and a matching file already exists in the cache,
        skip re-downloading. Set False to force a fresh download (e.g. if
        the link's target file changed).

    Returns
    -------
    str
        Local path to the downloaded checkpoint file.
    """
    from .utils import load_file_from_url  # local import: avoids a circular import at module load time

    direct_url = normalize_model_url(url)
    file_id = _gdrive_file_id(url)
    cache_dir = os.path.join(root_dir, _URL_CACHE_SUBDIR)
    os.makedirs(cache_dir, exist_ok=True)

    base_name = _cache_filename(url, file_id)
    # If a cached file with this base name (any extension) already exists, reuse it.
    if cache:
        for existing in os.listdir(cache_dir):
            if os.path.splitext(existing)[0] == base_name:
                path = os.path.join(cache_dir, existing)
                logger.info(f'[model_url] Using cached download: {path}')
                return path

    logger.info(f'[model_url] Downloading model from {direct_url} ...')
    downloaded_path = load_file_from_url(
        url=direct_url, model_dir=cache_dir, progress=True, file_name=None)

    # Google Drive's uc?export=download endpoint often serves the file
    # without a useful Content-Disposition filename hint, so the downloaded
    # file may not carry the right extension for universal_loader's
    # extension-based dispatch (.pth/.safetensors/.onnx/.ckpt). Rename it to
    # our deterministic cache name + whatever extension we can infer.
    if file_id:
        ext = os.path.splitext(downloaded_path)[1] or '.pth'
        target = os.path.join(cache_dir, base_name + ext)
        if downloaded_path != target:
            os.replace(downloaded_path, target)
            downloaded_path = target

    logger.info(f'[model_url] Model downloaded to {downloaded_path}')
    return downloaded_path
