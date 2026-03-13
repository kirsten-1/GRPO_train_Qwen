#!/usr/bin/env python3
"""Convert Stage1 JSON data into a local Hugging Face dataset directory for Open-R1 GRPO."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_INPUT = "/root/autodl-tmp/training_data/stage1_simple_math_train_ratio3.json"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/training_data/stage1_openr1_dataset"


def extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local Open-R1 dataset from Stage1 JSON.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to Stage1 JSON (list format).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory. It will contain train.jsonl for datasets.load_dataset(local_dir).",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, keep only this many samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for shuffle/subsample.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle records before optional truncation.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stage1_to_openr1_records(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, item in enumerate(samples):
        messages = item.get("messages", [])
        if len(messages) < 3:
            continue

        user_prompt = str(messages[1].get("content", "")).strip()
        assistant_text = str(messages[2].get("content", ""))
        solution = extract_tag_block(assistant_text, "answer")
        if not solution:
            solution = assistant_text.strip()

        if not user_prompt or not solution:
            continue

        records.append(
            {
                "id": item.get("id", f"stage1_{i}"),
                "problem": user_prompt,
                "solution": solution,
                "dataset": item.get("dataset", "unknown"),
                "task_type": item.get("task_type", "math"),
            }
        )
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    src_path = Path(args.input)
    out_dir = Path(args.output_dir)

    raw = load_json(src_path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list JSON in {src_path}, got {type(raw).__name__}")

    records = stage1_to_openr1_records(raw)
    if not records:
        raise ValueError("No valid records converted from input data.")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(records)

    if args.max_samples > 0:
        records = records[: args.max_samples]

    write_jsonl(records, out_dir / "train.jsonl")
    dataset_counts = Counter(r.get("dataset", "unknown") for r in records)
    save_json(
        {
            "input": str(src_path),
            "output_dir": str(out_dir),
            "num_records": len(records),
            "max_samples": args.max_samples,
            "shuffle": bool(args.shuffle),
            "seed": args.seed,
            "columns": ["id", "problem", "solution", "dataset", "task_type"],
            "dataset_counts": dict(dataset_counts),
        },
        out_dir / "meta.json",
    )

    print(f"Wrote Open-R1 dataset: {out_dir}/train.jsonl")
    print(f"Records: {len(records)}")
    for name in sorted(dataset_counts.keys()):
        pct = (dataset_counts[name] / len(records) * 100.0) if records else 0.0
        print(f"  - {name}: {dataset_counts[name]} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
