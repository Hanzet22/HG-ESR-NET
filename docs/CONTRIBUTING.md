# Contributing to HG-ESR-NET

:art: HG-ESR-NET needs your contributions. Any contributions are welcome, such as new features/architectures/typo fixes/suggestions/maintenance, *etc*. See [CONTRIBUTING.md](docs/CONTRIBUTING.md). All contributors are listed [here](../README.md#-based-on--credits).

This project builds on top of Real-ESRGAN's proven upscaling pipeline, extended with a universal architecture-detection layer. Individual strength is limited, so any kind of contribution is welcome, such as:

- New features
- New `archs2/` fallback architectures
- Bug fixes (loader, inference, ONNX path, etc.)
- Typo fixes
- Suggestions
- Maintenance
- Documents
- *etc*

## Workflow

1. Fork and pull the latest HG-ESR-NET repository
1. Checkout a new branch (do not use `main` branch for PRs)
1. Commit your changes
1. Create a PR

**Note**:

1. Please check the code style and linting
    1. The style configuration is specified in [setup.cfg](../setup.cfg)
    1. If you use VSCode, the settings are configured in [.vscode/settings.json](../.vscode/settings.json)
1. Strongly recommend using `pre-commit hook`. It will check your code style and linting before your commit.
    1. In the root path of project folder, run `pre-commit install`
    1. The pre-commit configuration is listed in [.pre-commit-config.yaml](../.pre-commit-config.yaml)
1. Better to [open a discussion](https://github.com/Hanzet22/HG-ESR-NET/discussions) before large changes.
    1. Welcome to discuss :sunglasses:. I will try my best to join the discussion.

## TODO List

:zero: Beyond the universal loader itself, here are some directions worth contributing to:

- [ ] Native TPU (v5e/v6e) inference support via `torch_xla` - static-shape bucketing, compile-graph friendly inference path
- [ ] Broaden `archs2/` fallback coverage for architectures Spandrel doesn't (yet) recognize
- [ ] Batched/queued inference mode for processing large folders more efficiently
- [ ] Controllable restoration strength (blend between input and full restoration)
- [ ] Improved ONNX export/inference parity with the PyTorch path

:one: There are also [several issues](https://github.com/Hanzet22/HG-ESR-NET/issues) that require helpers to improve. If you can help, please let me know :smile:

## Credit

This CONTRIBUTING guide's structure follows the original [Real-ESRGAN CONTRIBUTING.md](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/CONTRIBUTING.md) by Xintao Wang. See [README.md](../README.md#-based-on--credits) and [ATTRIBUTIONS.md](../ATTRIBUTIONS.md) for full upstream credit.
