"""
backends — TTS backend abstraction layer.

Usage:
    from backends import get_backend
    backend = get_backend()          # auto-detect platform
    backend = get_backend("mlx")     # explicit
"""

import platform

from backends.base import TTSBackend

__all__ = ["TTSBackend", "get_backend"]

_BACKENDS: dict[str, type[TTSBackend]] = {}


def _register_backends():
    """Populate the backend registry (lazy, runs once)."""
    if _BACKENDS:
        return
    from backends.mlx import MLXBackend
    from backends.vibevoice import VibeVoiceBackend
    _BACKENDS["mlx"] = MLXBackend
    _BACKENDS["vibevoice"] = VibeVoiceBackend


def _detect_backend() -> str:
    """Pick the best backend for the current platform."""
    machine = platform.machine().lower()

    # Apple Silicon → prefer MLX (optimized for this platform)
    if machine in ("arm64", "aarch64") and platform.system() == "Darwin":
        return "mlx"

    # Non-Apple-Silicon → try PyTorch-based backends
    for name, pkg in [("vibevoice", "vibevoice")]:
        try:
            __import__(pkg)
            return name
        except ImportError:
            continue

    raise RuntimeError(
        f"No TTS backend available for this platform ({platform.system()} {platform.machine()}).\n"
        "Install a backend:\n"
        "  Apple Silicon (MLX):  uv pip install -e '.[mlx]'\n"
        "  VibeVoice (PyTorch):  uv pip install -e '.[vibevoice]'"
    )


def get_backend(name: str | None = None) -> TTSBackend:
    """
    Return a TTSBackend instance.

    Parameters
    ----------
    name : str or None
        Backend name (e.g. "mlx", "indextts", "vibevoice").
        If *None*, auto-detect from platform.
    """
    _register_backends()

    if name is None:
        name = _detect_backend()

    name = name.lower()
    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS)) or "(none)"
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")

    return _BACKENDS[name]()
