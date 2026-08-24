"""
Universal architecture registry for the Real-ESRGAN compat layer.

Usage
-----
    from realesrgan.archs2.registry import ARCH2_REGISTRY

    # Hybrid mode (default): try auto-detect, fall back to manual name
    model, arch_name, confidence = ARCH2_REGISTRY.load(
        state_dict, arch_hint=args.arch  # None -> pure auto-detect
    )

    # Pure manual mode: force a specific arch, skip detection entirely
    model, arch_name, confidence = ARCH2_REGISTRY.load(
        state_dict, arch_hint=args.arch, force=True
    )

Adding a new architecture
--------------------------
Create a new file in realesrgan/archs2/, subclass BaseSRArch, implement
detect() and build(), then register it at the bottom of this file (or
call ARCH2_REGISTRY.register(YourArchClass) from anywhere that gets
imported before load() is called).
"""

import logging

logger = logging.getLogger('realesrgan.archs2')


class AmbiguousArchError(Exception):
    """Raised when two or more archs claim a state_dict with equal top confidence
    and no manual --arch hint was given to break the tie."""
    pass


class UnknownArchError(Exception):
    """Raised when no registered arch's detect() returns a positive score,
    and no manual --arch hint was given."""
    pass


class ArchRegistry:
    def __init__(self):
        self._archs = {}  # name -> class

    def register(self, arch_cls):
        name = arch_cls.ARCH_NAME
        if name in self._archs:
            logger.warning(f'[archs2] Overwriting already-registered arch "{name}"')
        self._archs[name] = arch_cls
        return arch_cls  # allows use as a decorator too

    def names(self):
        return sorted(self._archs.keys())

    def get(self, name):
        if name not in self._archs:
            raise UnknownArchError(
                f'Arch "{name}" is not registered. Available: {", ".join(self.names())}'
            )
        return self._archs[name]

    def detect_all(self, state_dict):
        """Run detect() for every registered arch. Returns list of (name, score)
        sorted descending by score."""
        sd = self._normalize(state_dict)
        scores = []
        for name, arch_cls in self._archs.items():
            try:
                score = arch_cls.detect(sd)
            except Exception as e:
                logger.debug(f'[archs2] detect() failed for "{name}": {e}')
                score = 0.0
            if score and score > 0.0:
                scores.append((name, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def load(self, state_dict, arch_hint=None, force=False, min_confidence=0.55,
              ambiguity_margin=0.05, **build_overrides):
        """Hybrid loader.

        Parameters
        ----------
        state_dict : dict
            Raw loaded checkpoint (any common wrapping is normalized internally).
        arch_hint : str or None
            Manual arch name. If None -> pure auto-detect.
            If given and force=False -> used only as a fallback / tiebreaker
            when auto-detect is ambiguous or below min_confidence.
            If given and force=True -> skip detection entirely, build directly.
        force : bool
            Skip auto-detection and use arch_hint directly. Requires arch_hint.
        min_confidence : float
            Below this score, auto-detect is considered "not confident enough"
            and arch_hint (if any) is required, else UnknownArchError.
        ambiguity_margin : float
            If the top-2 detected scores are within this margin of each other,
            treat as ambiguous and require arch_hint, else AmbiguousArchError.

        Returns
        -------
        (nn.Module, arch_name: str, confidence: float)
            confidence is 1.0 for forced/manual loads (no detection ran).
        """
        sd = self._normalize(state_dict)

        if force:
            if not arch_hint:
                raise ValueError('force=True requires arch_hint to be set.')
            arch_cls = self.get(arch_hint)
            model = arch_cls.build(sd, **build_overrides)
            return model, arch_hint, 1.0

        scores = self.detect_all(sd)

        if not scores:
            if arch_hint:
                logger.info(f'[archs2] Auto-detect found nothing; using manual --arch {arch_hint}')
                arch_cls = self.get(arch_hint)
                model = arch_cls.build(sd, **build_overrides)
                return model, arch_hint, 1.0
            raise UnknownArchError(
                'Could not auto-detect architecture and no --arch was given. '
                f'Registered archs: {", ".join(self.names())}'
            )

        top_name, top_score = scores[0]
        is_ambiguous = len(scores) > 1 and (top_score - scores[1][1]) < ambiguity_margin
        is_low_confidence = top_score < min_confidence

        if (is_ambiguous or is_low_confidence):
            if arch_hint:
                reason = 'ambiguous' if is_ambiguous else 'low-confidence'
                logger.info(
                    f'[archs2] Auto-detect {reason} (top candidates: '
                    f'{scores[:3]}); using manual --arch {arch_hint}'
                )
                arch_cls = self.get(arch_hint)
                model = arch_cls.build(sd, **build_overrides)
                return model, arch_hint, 1.0
            if is_ambiguous:
                raise AmbiguousArchError(
                    f'Multiple archs match with similar confidence: {scores[:3]}. '
                    'Pass --arch to disambiguate.'
                )
            raise UnknownArchError(
                f'Best auto-detect guess "{top_name}" only scored {top_score:.2f} '
                f'(< {min_confidence}). Pass --arch to force it or pick manually. '
                f'All candidates: {scores[:3]}'
            )

        # Confident, unambiguous auto-detect
        logger.info(f'[archs2] Auto-detected arch "{top_name}" (confidence {top_score:.2f})')
        arch_cls = self.get(top_name)
        model = arch_cls.build(sd, **build_overrides)
        return model, top_name, top_score

    @staticmethod
    def _normalize(state_dict):
        from .base import BaseSRArch
        return BaseSRArch.strip_prefix(state_dict)


# Singleton used across the codebase
ARCH2_REGISTRY = ArchRegistry()
