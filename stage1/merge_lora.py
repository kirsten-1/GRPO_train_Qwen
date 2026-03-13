#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model and save a standalone model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base-model", required=True, help="Base model path (local).")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter directory.")
    parser.add_argument("--output-dir", required=True, help="Merged model output directory.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Load dtype for merge.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="device_map for from_pretrained, e.g. auto/cpu/cuda:0.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--tokenizer-source",
        choices=["auto", "base", "adapter"],
        default="auto",
        help="Where to load tokenizer from. auto prefers adapter if tokenizer files are present.",
    )
    parser.add_argument(
        "--strict-base-check",
        action="store_true",
        help="Fail if adapter_config base_model_name_or_path mismatches --base-model.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def has_tokenizer_files(path: Path) -> bool:
    candidates = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    return any((path / name).exists() for name in candidates)


def read_adapter_base(adapter_path: Path) -> str:
    cfg_path = adapter_path / "adapter_config.json"
    if not cfg_path.exists():
        return ""
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return str(cfg.get("base_model_name_or_path", "")).strip()
    except Exception:
        return ""


def resolve_tokenizer_path(base_model: Path, adapter_path: Path, tokenizer_source: str) -> Path:
    if tokenizer_source == "base":
        return base_model
    if tokenizer_source == "adapter":
        return adapter_path
    return adapter_path if has_tokenizer_files(adapter_path) else base_model


def main() -> None:
    args = parse_args()
    base_model = Path(args.base_model)
    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    if not base_model.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_model}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    adapter_base = read_adapter_base(adapter_path)
    if adapter_base and Path(adapter_base) != base_model:
        msg = (
            "Adapter base_model_name_or_path does not match --base-model: "
            f"{adapter_base} vs {base_model}"
        )
        if args.strict_base_check:
            raise ValueError(msg)
        print(f"Warning: {msg}")

    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = resolve_dtype(args.dtype)
    print(f"[1/4] Loading base model from: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    print(f"[2/4] Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=args.local_files_only)
    print("[3/4] Merging adapter weights into base model")
    merged = model.merge_and_unload()
    print(f"[4/4] Saving merged model to: {output_dir}")
    merged.save_pretrained(str(output_dir), safe_serialization=args.safe_serialization)

    tokenizer_path = resolve_tokenizer_path(base_model, adapter_path, args.tokenizer_source)
    print(f"Saving tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer.save_pretrained(str(output_dir))

    del model
    del merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
