#!/usr/bin/env python3
"""Prepare Stage2 Open-R1 dataset with math-only default and optional Stage1 mix-in."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_INPUT = "/root/autodl-tmp/training_data/stage2_hard_math_train.json"
DEFAULT_STAGE1_INPUT = "/root/autodl-tmp/training_data/stage1_simple_math_train_ratio2.json"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/training_data/stage2_hf"


def extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local Open-R1 dataset from Stage2 JSON.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument(
        "--mix-stage1-input",
        type=str,
        default=DEFAULT_STAGE1_INPUT,
        help="Stage1 JSON used when --mix-stage1-ratio > 0.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, keep only first N after optional shuffle.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only-math",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only dataset=math from Stage2 input. Enabled by default.",
    )
    parser.add_argument(
        "--mix-stage1-ratio",
        type=float,
        default=0.0,
        help="Desired Stage1 ratio in final dataset (e.g. 0.10 means final set is ~90% stage2 + 10% stage1).",
    )
    parser.add_argument(
        "--mix-stage1-datasets",
        nargs="+",
        default=["gsm8k", "ape210k"],
        help="Allowed Stage1 datasets for mix-in.",
    )
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


def normalize_dataset_name(name: str) -> str:
    return str(name or "unknown").strip().lower()


def to_openr1_records(samples: List[Dict[str, Any]], id_prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(samples):
        msgs = item.get("messages", [])
        if len(msgs) < 3:
            continue
        problem = str(msgs[1].get("content", "")).strip()
        assistant = str(msgs[2].get("content", ""))
        solution = extract_tag_block(assistant, "answer")
        if not solution:
            solution = assistant.strip()
        if not problem or not solution:
            continue
        dataset_name = normalize_dataset_name(item.get("dataset", "unknown"))
        raw_id = item.get("id", f"{i}")
        out.append(
            {
                "id": f"{id_prefix}_{raw_id}",
                "problem": problem,
                "solution": solution,
                "dataset": dataset_name,
                "task_type": item.get("task_type", "math"),
            }
        )
    return out


def filter_records_by_dataset(records: List[Dict[str, Any]], allowed: Iterable[str]) -> List[Dict[str, Any]]:
    allow = {normalize_dataset_name(x) for x in allowed}
    return [r for r in records if normalize_dataset_name(r.get("dataset", "")) in allow]


def sample_stage1_mix(
    records: List[Dict[str, Any]],
    n_target: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if n_target <= 0:
        return []
    if n_target > len(records):
        raise ValueError(f"Requested stage1 mix {n_target}, but only {len(records)} samples available.")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[normalize_dataset_name(r.get("dataset", "unknown"))].append(r)

    total = len(records)
    desired: Dict[str, int] = {}
    remain = n_target
    remainders: List[tuple[float, str]] = []
    for ds, rows in groups.items():
        exact = n_target * len(rows) / total
        take = min(len(rows), int(exact))
        desired[ds] = take
        remain -= take
        remainders.append((exact - int(exact), ds))

    for _frac, ds in sorted(remainders, reverse=True):
        if remain <= 0:
            break
        cap = len(groups[ds]) - desired[ds]
        if cap <= 0:
            continue
        desired[ds] += 1
        remain -= 1

    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    for ds, rows in groups.items():
        n = desired.get(ds, 0)
        if n <= 0:
            continue
        picked.extend(rng.sample(rows, n) if len(rows) > n else rows)
    if len(picked) < n_target:
        used_ids = {x["id"] for x in picked}
        leftovers = [x for x in records if x["id"] not in used_ids]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: (n_target - len(picked))])
    return picked[:n_target]


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output_dir)

    raw = load_json(src)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list JSON in {src}, got {type(raw).__name__}")

    stage2_records = to_openr1_records(raw, id_prefix="stage2")
    if args.only_math:
        stage2_records = filter_records_by_dataset(stage2_records, ["math"])
    if not stage2_records:
        raise ValueError("No valid Stage2 records after filtering.")

    records = list(stage2_records)
    stage1_mix_records: List[Dict[str, Any]] = []
    if args.mix_stage1_ratio > 0:
        if not (0.0 < args.mix_stage1_ratio < 1.0):
            raise ValueError("--mix-stage1-ratio must be in (0, 1).")
        stage1_path = Path(args.mix_stage1_input)
        stage1_raw = load_json(stage1_path)
        if not isinstance(stage1_raw, list):
            raise ValueError(f"Expected list JSON in {stage1_path}, got {type(stage1_raw).__name__}")
        stage1_all = to_openr1_records(stage1_raw, id_prefix="stage1mix")
        stage1_pool = filter_records_by_dataset(stage1_all, args.mix_stage1_datasets)
        if not stage1_pool:
            raise ValueError(
                "No Stage1 records matched --mix-stage1-datasets. "
                f"Got filters={args.mix_stage1_datasets}, total={len(stage1_all)}."
            )

        mix_target = int(round(len(stage2_records) * args.mix_stage1_ratio / (1.0 - args.mix_stage1_ratio)))
        mix_target = max(1, mix_target)
        stage1_mix_records = sample_stage1_mix(stage1_pool, mix_target, seed=args.seed + 17)
        records.extend(stage1_mix_records)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(records)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No valid records generated for Stage2.")

    write_jsonl(records, dst / "train.jsonl")
    counts = Counter(x.get("dataset", "unknown") for x in records)
    save_json(
        {
            "input": str(src),
            "mix_stage1_input": str(args.mix_stage1_input),
            "output_dir": str(dst),
            "num_records": len(records),
            "num_stage2_records": len(stage2_records),
            "num_stage1_mix_records": len(stage1_mix_records),
            "dataset_counts": dict(counts),
            "columns": ["id", "problem", "solution", "dataset", "task_type"],
            "seed": args.seed,
            "shuffle": bool(args.shuffle),
            "max_samples": args.max_samples,
            "only_math": bool(args.only_math),
            "mix_stage1_ratio": args.mix_stage1_ratio,
            "mix_stage1_datasets": [normalize_dataset_name(x) for x in args.mix_stage1_datasets],
        },
        dst / "meta.json",
    )
    print(f"Wrote {len(records)} records to {dst}/train.jsonl")
    for k in sorted(counts.keys()):
        pct = counts[k] / len(records) * 100.0
        print(f"  - {k}: {counts[k]} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
