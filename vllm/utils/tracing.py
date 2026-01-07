# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Platform-agnostic GPU tracing utilities.

This module provides a unified interface for GPU profiling that works
on both NVIDIA (nvtx) and AMD (roctx) platforms. It automatically detects
the available platform and uses the appropriate tracing library.

The API mirrors nvtx v0 which has feature parity with roctx:
- range_push(message) / range_pop() - Nested range markers
- mark(message) - Point markers
- annotate(message) - Context manager for scoped ranges
"""

from contextlib import contextmanager
from typing import Any

# Detect platform and import appropriate tracing library
_TRACING_BACKEND: str | None = None
_range_push = None
_range_pop = None
_mark = None
_annotate = None


def _init_tracing_backend():
    """Initialize the tracing backend based on available platform."""
    global _TRACING_BACKEND, _range_push, _range_pop, _mark, _annotate

    if _TRACING_BACKEND is not None:
        return  # Already initialized

    # Try to detect platform using vLLM's platform detection
    try:
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            # Try roctx for ROCm
            try:
                from roctx import range_pop, range_push
                # roctx doesn't have mark, we'll implement it as a zero-length range
                _range_push = range_push
                _range_pop = range_pop
                _mark = lambda msg: (range_push(msg), range_pop())
                _TRACING_BACKEND = "roctx"
                return
            except ImportError:
                pass

            # Fallback: try torch.cuda.nvtx (works on ROCm via HIP)
            try:
                import torch.cuda.nvtx as nvtx
                _range_push = nvtx.range_push
                _range_pop = nvtx.range_pop
                _mark = lambda msg: (nvtx.range_push(msg), nvtx.range_pop())
                _TRACING_BACKEND = "torch_nvtx_rocm"
                return
            except (ImportError, AttributeError):
                pass

        elif current_platform.is_cuda():
            # Try standalone nvtx package first (more features)
            try:
                import nvtx
                _range_push = nvtx.push_range
                _range_pop = nvtx.pop_range
                _mark = nvtx.mark
                _annotate = nvtx.annotate
                _TRACING_BACKEND = "nvtx"
                return
            except ImportError:
                pass

            # Fallback to torch.cuda.nvtx
            try:
                import torch.cuda.nvtx as nvtx
                _range_push = nvtx.range_push
                _range_pop = nvtx.range_pop
                _mark = lambda msg: (nvtx.range_push(msg), nvtx.range_pop())
                _TRACING_BACKEND = "torch_nvtx"
                return
            except (ImportError, AttributeError):
                pass

    except ImportError:
        # Platform detection not available, try backends directly
        pass

    # Direct fallback attempts without platform detection
    try:
        from roctx import range_pop, range_push
        _range_push = range_push
        _range_pop = range_pop
        _mark = lambda msg: (range_push(msg), range_pop())
        _TRACING_BACKEND = "roctx"
        return
    except ImportError:
        pass

    try:
        import nvtx
        _range_push = nvtx.push_range
        _range_pop = nvtx.pop_range
        _mark = nvtx.mark
        _annotate = nvtx.annotate
        _TRACING_BACKEND = "nvtx"
        return
    except ImportError:
        pass

    try:
        import torch.cuda.nvtx as nvtx
        _range_push = nvtx.range_push
        _range_pop = nvtx.range_pop
        _mark = lambda msg: (nvtx.range_push(msg), nvtx.range_pop())
        _TRACING_BACKEND = "torch_nvtx"
        return
    except (ImportError, AttributeError):
        pass

    # No tracing backend available - use no-ops
    _TRACING_BACKEND = "none"
    _range_push = lambda msg: None
    _range_pop = lambda: None
    _mark = lambda msg: None


def get_tracing_backend() -> str:
    """Return the name of the active tracing backend.

    Returns:
        One of: "nvtx", "roctx", "torch_nvtx", "torch_nvtx_rocm", "none"
    """
    _init_tracing_backend()
    return _TRACING_BACKEND


def is_tracing_available() -> bool:
    """Check if GPU tracing is available on the current platform."""
    return get_tracing_backend() != "none"


def range_push(message: str) -> Any:
    """Push a named range onto the tracing stack.

    Args:
        message: The name/label for the range.

    Returns:
        A range ID (backend-specific) or None.
    """
    _init_tracing_backend()
    return _range_push(message)


def range_pop() -> None:
    """Pop the most recent range from the tracing stack."""
    _init_tracing_backend()
    _range_pop()


def mark(message: str) -> None:
    """Create an instantaneous marker in the trace.

    Args:
        message: The label for the marker.
    """
    _init_tracing_backend()
    _mark(message)


@contextmanager
def annotate(message: str, **kwargs):
    """Context manager for creating a traced range.

    This is the preferred way to trace a code region as it ensures
    proper push/pop pairing even if exceptions occur.

    Args:
        message: The name/label for the range.
        **kwargs: Additional backend-specific options (e.g., color for nvtx).

    Yields:
        None

    Example:
        with annotate("forward_pass"):
            model(inputs)
    """
    _init_tracing_backend()

    # Use native annotate if available (nvtx package)
    if _annotate is not None:
        with _annotate(message, **kwargs):
            yield
    else:
        # Fallback to push/pop
        range_push(message)
        try:
            yield
        finally:
            range_pop()


class NullAnnotate:
    """A null context manager that does nothing.

    Used as a no-op replacement when tracing is disabled.
    """

    def __init__(self, message: str = "", **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def get_annotate_func():
    """Get the appropriate annotate function based on platform.

    Returns:
        A callable that returns a context manager for tracing ranges.
        Returns NullAnnotate if tracing is not available.
    """
    _init_tracing_backend()

    if _TRACING_BACKEND == "nvtx" and _annotate is not None:
        return _annotate
    elif _TRACING_BACKEND != "none":
        return annotate
    else:
        return NullAnnotate

