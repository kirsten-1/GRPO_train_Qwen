#!/usr/bin/env python3
"""Prepare Stage3 Open-R1 dataset (code + math mix) with tag normalization."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT = "/root/autodl-tmp/training_data/stage3_code_train.json"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/training_data/stage3_hf"


def normalize_tags(text: str) -> str:
    """Convert <reasoning>...</reasoning> to <think>...</think> for consistency."""
    text = re.sub(r"<reasoning>", "<think>", text)
    text = re.sub(r"</reasoning>", "</think>", text)
    return text


def normalize_system_prompt(text: str) -> str:
    """Replace system prompt to use <think>/<answer> format."""
    text = normalize_tags(text)
    return text


def extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stage3 dataset from code+math JSON.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_openr1_records(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(samples):
        msgs = item.get("messages", [])
        if len(msgs) < 3:
            continue
        problem = str(msgs[1].get("content", "")).strip()
        assistant = normalize_tags(str(msgs[2].get("content", "")))
        solution = extract_tag_block(assistant, "answer")
        if not solution:
            solution = assistant.strip()
        if not problem or not solution:
            continue
        dataset_name = str(item.get("dataset", "unknown")).strip().lower()
        raw_id = item.get("id", f"stage3_{i}")
        out.append(
            {
                "id": f"stage3_{raw_id}",
                "problem": problem,
                "solution": solution,
                "dataset": dataset_name,
                "task_type": item.get("task_type", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output_dir)

    raw = load_json(src)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list JSON in {src}, got {type(raw).__name__}")

    records = to_openr1_records(raw)
    if not records:
        raise ValueError("No valid records generated.")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(records)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    write_jsonl(records, dst / "train.jsonl")
    counts = Counter(x.get("dataset", "unknown") for x in records)
    task_counts = Counter(x.get("task_type", "unknown") for x in records)

    # Verify tag normalization
    reasoning_count = sum(1 for r in records if "<reasoning>" in r.get("solution", ""))
    think_count = sum(1 for r in records if "<think>" in r.get("solution", ""))

    save_json(
        {
            "input": str(src),
            "output_dir": str(dst),
            "num_records": len(records),
            "dataset_counts": dict(counts),
            "task_type_counts": dict(task_counts),
            "columns": ["id", "problem", "solution", "dataset", "task_type", "difficulty"],
            "seed": args.seed,
            "shuffle": bool(args.shuffle),
            "max_samples": args.max_samples,
            "tag_check": {
                "remaining_reasoning_tags": reasoning_count,
                "think_tags": think_count,
            },
        },
        dst / "meta.json",
    )
    print(f"Wrote {len(records)} records to {dst}/train.jsonl")
    for k in sorted(counts.keys()):
        pct = counts[k] / len(records) * 100.0
        print(f"  dataset {k}: {counts[k]} ({pct:.1f}%)")
    for k in sorted(task_counts.keys()):
        pct = task_counts[k] / len(records) * 100.0
        print(f"  task_type {k}: {task_counts[k]} ({pct:.1f}%)")
    print(f"  Tag check: <reasoning> remaining={reasoning_count}, <think>={think_count}")


if __name__ == "__main__":
    main()
