"""
Quick sanity check for the universal architecture compat layer.

Run this FIRST after `pip install -r requirements.txt`, before trusting
the loader on your real models. It does three things:

  1. Import check       - confirms torch/spandrel/etc actually installed
  2. Registry check      - confirms archs2/ fallback registers correctly
  3. Load check (optional) - if you pass a model path, actually loads it
                             and reports which backend/arch handled it

Usage
-----
    # just check the environment is sane
    python test_universal_loader.py

    # actually try loading one of your models
    python test_universal_loader.py --model_path weights/whatever.pth
    python test_universal_loader.py --model_path weights/whatever.pth --arch rrdbnet
"""

import argparse
import sys


def check_imports():
    print('=== 1. Import check ===')
    results = {}
    for name in ('torch', 'spandrel', 'spandrel_extra_arches', 'safetensors', 'onnxruntime', 'openmodeldb', 'cv2'):
        try:
            mod = __import__(name)
            ver = getattr(mod, '__version__', 'unknown')
            print(f'  OK    {name:<24} {ver}')
            results[name] = True
        except ImportError as e:
            required = name in ('torch', 'spandrel')
            tag = 'MISSING (required)' if required else 'missing (optional)'
            print(f'  {"FAIL" if required else "warn":<5} {name:<24} {tag} - {e}')
            results[name] = False

    if not results.get('torch'):
        print('\ntorch is required. Install it first: pip install torch>=2.2.0')
        sys.exit(1)
    if not results.get('spandrel'):
        print('\nspandrel is missing - only the archs2/ fallback (rrdbnet, srvgg_compact, '
              'realcugan) will work. Install it for 25+ architectures: pip install spandrel')
    return results


def check_registry():
    print('\n=== 2. archs2/ fallback registry check ===')
    try:
        from realesrgan.archs2.registry import ARCH2_REGISTRY
        from realesrgan.archs2 import rrdbnet_arch, srvgg_arch, realcugan_arch  # noqa: F401
        names = ARCH2_REGISTRY.names()
        if not names:
            print('  FAIL  registry is empty - archs did not self-register')
            sys.exit(1)
        print(f'  OK    {len(names)} archs registered: {", ".join(names)}')
    except Exception as e:
        print(f'  FAIL  could not import archs2 registry: {e}')
        sys.exit(1)


def check_cuda():
    print('\n=== 3. Device check ===')
    import torch
    if torch.cuda.is_available():
        print(f'  OK    CUDA available - {torch.cuda.get_device_name(0)}')
    else:
        print('  warn  CUDA not available - will run on CPU (slow, but fine for testing)')


def try_load_model(model_path, arch_hint=None):
    print(f'\n=== 4. Load test: {model_path} ===')
    from realesrgan.universal_loader import load_sr_model
    try:
        model, meta = load_sr_model(model_path, arch_hint=arch_hint, device='cpu')
    except Exception as e:
        print(f'  FAIL  could not load model: {e}')
        sys.exit(1)

    print(f'  OK    backend={meta["backend"]}  arch={meta["arch"]}  '
          f'scale={meta["scale"]}  confidence={meta["confidence"]:.2f}')

    # count params as a basic "did this actually build a real network" sanity check
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  OK    model built with {n_params:,} parameters')

    # one dummy forward pass to catch shape mismatches early
    import torch
    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.rand(1, 3, 64, 64)
            out = model(dummy)
        print(f'  OK    dummy forward pass: input {tuple(dummy.shape)} -> output {tuple(out.shape)}')
    except Exception as e:
        print(f'  FAIL  forward pass failed - architecture was detected/built but is likely '
              f'wrong for this checkpoint: {e}')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_path', type=str, default=None,
                         help='Optional: path to a .pth/.ckpt/.pt/.safetensors file to actually load and test.')
    parser.add_argument('--arch', type=str, default=None,
                         help='Optional arch hint, forwarded to the archs2/ fallback if spandrel misses.')
    args = parser.parse_args()

    check_imports()
    check_registry()
    check_cuda()

    if args.model_path:
        try_load_model(args.model_path, arch_hint=args.arch)
        print('\nAll checks passed. This model is safe to use with inference_realesrgan.py --model_path')
    else:
        print('\nEnvironment checks passed. Run again with --model_path <file> to test an actual model.')
