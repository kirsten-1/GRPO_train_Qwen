#!/usr/bin/env python3
"""Compute recommended max_steps for Stage2 continuation from a checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calculate Stage2 max_steps for incremental epochs.")
    p.add_argument("--dataset-jsonl", default="/root/autodl-tmp/training_data/stage2_hf/train.jsonl")
    p.add_argument("--resume-checkpoint", required=True, help="Path like .../checkpoint-22000")
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument(
        "--mode",
        choices=["observed", "classic_hf"],
        default="observed",
        help="observed: steps_per_epoch = dataset_samples (matches observed Stage1 run). "
        "classic_hf: ceil(samples/(bs*ga*world_size)).",
    )
    p.add_argument("--epochs-to-run", type=int, default=1, help="Additional Stage2 epochs to run.")
    return p.parse_args()


def count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def parse_resume_step(ckpt: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", ckpt.as_posix())
    if m:
        return int(m.group(1))
    state = ckpt / "trainer_state.json"
    if state.exists():
        with state.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return int(obj.get("global_step", 0))
    raise ValueError(f"Cannot infer global step from checkpoint path: {ckpt}")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_jsonl)
    ckpt = Path(args.resume_checkpoint)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset jsonl: {dataset_path}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint path: {ckpt}")

    n_samples = count_jsonl(dataset_path)
    if args.mode == "observed":
        steps_per_epoch = n_samples
    else:
        denom = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
        steps_per_epoch = int(math.ceil(n_samples / max(1, denom)))
    resume_step = parse_resume_step(ckpt)
    target_max_steps = resume_step + steps_per_epoch * max(1, args.epochs_to_run)

    print(f"mode={args.mode}")
    print(f"dataset_samples={n_samples}")
    print(f"steps_per_epoch={steps_per_epoch}")
    print(f"resume_step={resume_step}")
    print(f"recommended_max_steps={target_max_steps}")


if __name__ == "__main__":
    main()
