#!/usr/bin/env python3
"""Global compatibility shims loaded automatically by Python site import.

This file is intentionally lightweight and safe to import in subprocesses.
"""

from __future__ import annotations

import logging
from typing import Any, Optional


def _patch_vllm_guided_decoding() -> None:
    try:
        import vllm.sampling_params as sp  # type: ignore
    except Exception:
        return

    if hasattr(sp, "GuidedDecodingParams"):
        return

    class GuidedDecodingParams:
        def __init__(self, backend: str = "outlines", regex: Optional[str] = None, **kwargs: Any):
            self.backend = backend
            self.regex = regex
            self.kwargs = kwargs

    sp.GuidedDecodingParams = GuidedDecodingParams  # type: ignore[attr-defined]


def _patch_vllm_get_open_port() -> None:
    try:
        import vllm.utils as vutils  # type: ignore
    except Exception:
        return

    if hasattr(vutils, "get_open_port"):
        return

    get_open_port = None
    try:
        from vllm.utils.network_utils import get_open_port as _get_open_port  # type: ignore
        get_open_port = _get_open_port
    except Exception:
        pass
    if get_open_port is None:
        try:
            from vllm.v1.utils import get_open_port as _get_open_port  # type: ignore
            get_open_port = _get_open_port
        except Exception:
            pass
    if get_open_port is not None:
        vutils.get_open_port = get_open_port  # type: ignore[attr-defined]


def _suppress_noisy_loggers() -> None:
    # Third-party deprecation spam from reward normalization; keep training logs readable.
    for name in (
        "latex2sympy2_extended.math_normalization",
        "math_verify.math_normalization",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


_patch_vllm_guided_decoding()
_patch_vllm_get_open_port()
_suppress_noisy_loggers()
