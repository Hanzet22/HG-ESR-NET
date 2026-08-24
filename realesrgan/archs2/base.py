"""
Base interface for every architecture plugged into the universal registry.

Design goal: every arch module (RRDBNet, SwinIR, HAT, RealCUGAN, SPAN, ...)
exposes the same three things so the loader/detector/inferencer never need
to know arch-specific details:

    1. ARCH_NAME        - unique string id, e.g. "rrdbnet"
    2. detect(state_dict) -> float score in [0, 1]
           0.0   = definitely not this arch
           1.0   = definitely this arch (exact key/shape match)
           (0,1) = partial match, used for tie-breaking when several
                   archs claim a state_dict
    3. build(state_dict) -> nn.Module
           Construct the arch with hyperparameters *inferred from the
           state_dict itself* (channel counts, block counts, scale, etc)
           and load the weights, so the caller never has to pass in a
           config file for standard community checkpoints.

Keep detect() cheap: it runs against every registered arch for every model
file, so it should only look at key names / tensor shapes, never actually
build a network.
"""

from abc import ABC, abstractmethod


class BaseSRArch(ABC):
    """Abstract base every arch adapter in archs2/ must implement."""

    ARCH_NAME: str = "base"

    # Rough family tag, purely informational (shown in logs / --list-archs).
    FAMILY: str = "unknown"

    @classmethod
    @abstractmethod
    def detect(cls, state_dict: dict) -> float:
        """Return a confidence score in [0, 1] that state_dict is this arch."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build(cls, state_dict: dict, **overrides):
        """Build and return an nn.Module with weights loaded from state_dict.

        `overrides` lets a manual --arch-opt flag override any inferred
        hyperparameter (e.g. num_block=23) without touching detection logic.
        """
        raise NotImplementedError

    @classmethod
    def strip_prefix(cls, state_dict: dict) -> dict:
        """Normalize common checkpoint wrapping before detection/building.

        Handles the usual suspects seen across OpenModelDB / HuggingFace
        uploads: {'params_ema': ...}, {'params': ...}, {'state_dict': ...},
        plain nn.DataParallel 'module.' prefixes, etc.
        """
        sd = state_dict
        for key in ('params_ema', 'params', 'state_dict', 'model', 'net', 'model_state_dict'):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
        if isinstance(sd, dict) and any(k.startswith('module.') for k in sd.keys()):
            sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
        return sd
