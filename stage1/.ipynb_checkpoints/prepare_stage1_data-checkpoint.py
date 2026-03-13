#!/usr/bin/env python3
"""
Prepare Stage1 math training data with configurable Ape210K:GSM8K ratio.

Default ratio is 3:1 (Ape:GSM), which makes GSM8K about 25%.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_PROCESSED_DIR = Path("/root/autodl-tmp/processed_datasets")
DEFAULT_OUTPUT_PATH = Path("/root/autodl-tmp/training_data/stage1_simple_math_train_ratio3.json")


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}, got {type(data)}")
    return data


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def rebalance_samples(
    gsm_samples: List[Dict[str, Any]],
    ape_samples: List[Dict[str, Any]],
    ape_to_gsm_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if ape_to_gsm_ratio <= 0:
        raise ValueError(f"ape_to_gsm_ratio must be > 0, got {ape_to_gsm_ratio}")

    rng = random.Random(seed)
    target_ape_count = int(round(len(gsm_samples) * ape_to_gsm_ratio))
    if target_ape_count <= 0:
        raise ValueError("Target Ape210K sample count is <= 0. Check ratio and gsm dataset size.")

    if len(ape_samples) >= target_ape_count:
        selected_ape = rng.sample(ape_samples, target_ape_count)
    else:
        selected_ape = list(ape_samples)

    mixed = list(gsm_samples) + selected_ape
    rng.shuffle(mixed)
    return mixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stage1 training set with custom Ape:GSM ratio.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--gsm-file", type=str, default="gsm8k_train.json")
    parser.add_argument("--ape-file", type=str, default="ape210k_train.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--ape-to-gsm-ratio", type=float, default=3.0, help="Ape210K:GSM8K ratio.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gsm_path = args.processed_dir / args.gsm_file
    ape_path = args.processed_dir / args.ape_file
    if not gsm_path.exists():
        raise FileNotFoundError(f"Missing GSM8K file: {gsm_path}")
    if not ape_path.exists():
        raise FileNotFoundError(f"Missing Ape210K file: {ape_path}")

    gsm_samples = load_json(gsm_path)
    ape_samples = load_json(ape_path)
    mixed = rebalance_samples(
        gsm_samples=gsm_samples,
        ape_samples=ape_samples,
        ape_to_gsm_ratio=args.ape_to_gsm_ratio,
        seed=args.seed,
    )
    save_json(args.output, mixed)

    counts = Counter(x.get("dataset", "unknown") for x in mixed)
    total = len(mixed)
    print("Stage1 data prepared")
    print(f"Output: {args.output}")
    print(f"Total samples: {total}")
    for name in sorted(counts.keys()):
        pct = (counts[name] / total * 100.0) if total else 0.0
        print(f"  - {name}: {counts[name]} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
