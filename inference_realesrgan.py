import argparse
import cv2
import glob
import os

from realesrgan import RealESRGANer

# Built-in official model registry - kept ONLY for the convenience -n flag
# (download-by-name for the original 6 official checkpoints). Any other
# model, from OpenModelDB/HuggingFace/anywhere, is loaded via --model_path
# through the universal loader (spandrel primary, archs2/ fallback) and
# does NOT need an entry here.
_OFFICIAL_MODELS = {
    'RealESRGAN_x4plus': (4, ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']),
    'RealESRNet_x4plus': (4, ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']),
    'RealESRGAN_x4plus_anime_6B': (4, ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']),
    'RealESRGAN_x2plus': (2, ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']),
    'realesr-animevideov3': (4, ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']),
    'realesr-general-x4v3': (4, [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
    ]),
}


def main():
    """Inference demo for Real-ESRGAN - universal architecture edition.

    Two ways to pick a model:
      -n / --model_name   : one of the 6 official checkpoints (auto-downloaded)
      --model_path         : ANY local .pth/.ckpt/.pt/.safetensors file, from
                              OpenModelDB, HuggingFace, or your own training.
                              Architecture is auto-detected (spandrel, then
                              the archs2/ fallback). Use --arch to hint/force
                              it if auto-detection is ambiguous.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default=None,
        help=('Official model name (auto-downloaded): ' + ' | '.join(_OFFICIAL_MODELS.keys()) +
              '. Ignored if --model_path is given.'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help=('Path to ANY local model checkpoint (.pth / .ckpt / .pt / .safetensors) - '
              'e.g. a model downloaded from OpenModelDB or HuggingFace. Architecture is '
              'auto-detected. Overrides --model_name if both are given.'))
    parser.add_argument(
        '--arch',
        type=str,
        default=None,
        help=('Optional architecture hint/override, used only if --model_path is given AND '
              'spandrel cannot auto-detect the architecture (falls back to the archs2/ '
              'registry: rrdbnet | srvgg_compact | realcugan). Run with --list_archs to see '
              'all archs2/ fallback names. For the 25+ architectures spandrel supports '
              'natively (SPAN, OmniSR, SAFMN, HAT, DAT, SwinIR, etc.), this is not needed - '
              'spandrel detects them automatically.'))
    parser.add_argument(
        '--scale',
        type=int,
        default=None,
        help=('Network upsampling scale (2, 3, 4...). Usually auto-detected from the checkpoint '
              'when using --model_path; only needed if detection fails.'))
    parser.add_argument('--list_archs', action='store_true',
                         help='List architecture names known to the archs2/ fallback registry and exit.')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    if args.list_archs:
        from realesrgan.archs2.registry import ARCH2_REGISTRY
        from realesrgan.archs2 import rrdbnet_arch, srvgg_arch, realcugan_arch  # noqa: F401  (trigger registration)
        print('archs2/ fallback registry (used only when spandrel does not recognize a checkpoint):')
        for name in ARCH2_REGISTRY.names():
            print(f'  - {name}')
        print('\nFor the full architecture list (SPAN, OmniSR, SAFMN, HAT, DAT, SwinIR, Swin2SR, '
              'SRFormer, RGT, DRCT, ATD, RealCUGAN, PLKSR, RestoreFormer, RetinexFormer, SCUNet, '
              'ESRGAN/RRDBNet, SRVGGCompact, and more) see: pip show spandrel spandrel_extra_arches')
        return

    dni_weight = None

    # --- Determine model_path + scale ---
    if args.model_path is not None:
        # Universal path: any checkpoint, any of the 25+ archs (spandrel) or
        # the archs2/ fallback. RealESRGANer(model=None, ...) triggers
        # universal_loader internally.
        model_path = args.model_path
        netscale = args.scale  # may be None -> inferred by universal_loader from the checkpoint
        model = None
    else:
        # Official-name path (backward compatible with the original script)
        model_name = (args.model_name or 'RealESRGAN_x4plus').split('.')[0]
        if model_name not in _OFFICIAL_MODELS:
            raise ValueError(
                f'Unknown --model_name "{model_name}" and no --model_path given. '
                f'Official names: {", ".join(_OFFICIAL_MODELS.keys())}. '
                f'For any other model (OpenModelDB/HuggingFace/custom), use --model_path instead.'
            )
        netscale, file_url = _OFFICIAL_MODELS[model_name]
        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            from realesrgan.utils import load_file_from_url
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        if model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

        # official names still resolve to a concrete arch via universal_loader too
        # (model=None), so we don't need to hand-construct RRDBNet/SRVGGNetCompact here.
        model = None

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        arch=args.arch,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    if hasattr(upsampler, '_load_meta'):
        m = upsampler._load_meta
        print(f'Loaded model - backend: {m["backend"]}, arch: {m["arch"]}, scale: {upsampler.scale}, '
              f'precision: {"fp32" if args.fp32 else "fp16"}')

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'  Skipping {os.path.basename(path)}: not readable as an image by OpenCV '
                  f'(not an image file, or a format cv2 does not support - e.g. video files, '
                  f'some RAW formats, corrupted files).')
            continue
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)


if __name__ == '__main__':
    main()
