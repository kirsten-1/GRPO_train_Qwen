#!/usr/bin/env python3
"""Summarize Stage2 training dynamics from trainer_state.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize loss/reward/KL/LR from trainer_state.json.")
    p.add_argument(
        "--trainer-state",
        default="/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090/trainer_state.json",
    )
    p.add_argument("--output", default="")
    return p.parse_args()


def collect(hist: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for x in hist:
        v = x.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"count": 0.0}
    return {
        "count": float(len(vals)),
        "first": vals[0],
        "last": vals[-1],
        "min": min(vals),
        "max": max(vals),
        "mean": mean(vals),
    }


def main() -> None:
    args = parse_args()
    path = Path(args.trainer_state)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    hist = state.get("log_history", [])
    if not isinstance(hist, list):
        raise ValueError("log_history missing or invalid")

    out = {
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "loss": stats(collect(hist, "loss")),
        "reward": stats(collect(hist, "reward")),
        "kl": stats(collect(hist, "kl")),
        "learning_rate": stats(collect(hist, "learning_rate")),
        "kl_sum": sum(collect(hist, "kl")),
    }

    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
