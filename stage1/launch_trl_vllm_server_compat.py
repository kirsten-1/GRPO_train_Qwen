#!/usr/bin/env python3
"""Launch TRL vLLM server with compatibility shims for newer vLLM APIs."""

from __future__ import annotations

import argparse
import inspect
from typing import Any, Optional


def patch_vllm_symbols() -> tuple[Any, bool]:
    import vllm.sampling_params as sp  # type: ignore
    import vllm.utils as vutils  # type: ignore

    needs_sampling_patch = False

    if not hasattr(sp, "GuidedDecodingParams"):
        class GuidedDecodingParams:
            def __init__(self, backend: str = "outlines", regex: Optional[str] = None, **kwargs: Any):
                self.backend = backend
                self.regex = regex
                self.kwargs = kwargs

        sp.GuidedDecodingParams = GuidedDecodingParams  # type: ignore[attr-defined]

    if "guided_decoding" not in inspect.signature(sp.SamplingParams).parameters:
        needs_sampling_patch = True

    if not hasattr(vutils, "get_open_port"):
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
        if get_open_port is None:
            raise ImportError("Cannot locate get_open_port in current vLLM installation.")
        vutils.get_open_port = get_open_port  # type: ignore[attr-defined]

    return sp, needs_sampling_patch


def patch_trl_vllm_serve_sampling_params(vserve: Any, sp: Any) -> None:
    if getattr(vserve, "_stage1_sampling_patch_applied", False):
        return

    orig_sampling_params = vserve.SamplingParams
    if "guided_decoding" in inspect.signature(orig_sampling_params).parameters:
        return

    class SamplingParamsCompat:
        def __new__(cls, *args: Any, guided_decoding: Any = None, **kwargs: Any):
            if guided_decoding is not None and "structured_outputs" not in kwargs:
                regex = getattr(guided_decoding, "regex", None)
                if regex and hasattr(sp, "StructuredOutputsParams"):
                    kwargs["structured_outputs"] = sp.StructuredOutputsParams(regex=regex)
            return orig_sampling_params(*args, **kwargs)

    vserve.SamplingParams = SamplingParamsCompat  # type: ignore[assignment]
    vserve._stage1_sampling_patch_applied = True  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 TRL vLLM server compat launcher.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--kv_cache_dtype", default="auto")
    parser.add_argument("--log_level", default="info")
    parser.add_argument("--enable_prefix_caching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enforce_eager", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sp, needs_sampling_patch = patch_vllm_symbols()

    from trl.scripts import vllm_serve as vserve

    if needs_sampling_patch:
        patch_trl_vllm_serve_sampling_params(vserve, sp)

    script_args = vserve.ScriptArguments(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        host=args.host,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=(False if args.enforce_eager is None else args.enforce_eager),
        kv_cache_dtype=args.kv_cache_dtype,
        log_level=args.log_level,
    )
    vserve.main(script_args)


if __name__ == "__main__":
    main()
