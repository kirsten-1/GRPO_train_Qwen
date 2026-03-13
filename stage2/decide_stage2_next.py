#!/usr/bin/env python3
"""Decision helper: whether to continue Stage2 / move to Stage3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decide next step from Stage2 eval outputs.")
    p.add_argument("--stage1-eval", required=True, help="Path to Stage1 eval_results.json")
    p.add_argument("--stage2-eval", required=True, help="Path to Stage2 eval_results.json")
    p.add_argument("--gsm8k-max-drop", type=float, default=5.0, help="Max allowed drop in percentage points.")
    p.add_argument("--ape210k-max-drop", type=float, default=5.0, help="Max allowed drop in percentage points.")
    p.add_argument("--output", default="")
    return p.parse_args()


def get_pct(metrics: dict, key: str) -> float:
    return float(metrics.get(key, 0.0)) * 100.0


def main() -> None:
    args = parse_args()
    s1 = json.loads(Path(args.stage1_eval).read_text(encoding="utf-8"))
    s2 = json.loads(Path(args.stage2_eval).read_text(encoding="utf-8"))
    m1 = s1.get("metrics", {})
    m2 = s2.get("metrics", {})

    gsm1 = get_pct(m1, "gsm8k_accuracy")
    ape1 = get_pct(m1, "ape210k_accuracy")
    math1 = get_pct(m1, "math_accuracy")
    cmath1 = get_pct(m1, "cmath_accuracy")

    gsm2 = get_pct(m2, "gsm8k_accuracy")
    ape2 = get_pct(m2, "ape210k_accuracy")
    math2 = get_pct(m2, "math_accuracy")
    cmath2 = get_pct(m2, "cmath_accuracy")

    gsm_drop = gsm1 - gsm2
    ape_drop = ape1 - ape2
    math_gain = math2 - math1
    cmath_gain = cmath2 - cmath1

    retain_ok = (gsm_drop <= args.gsm8k_max_drop) and (ape_drop <= args.ape210k_max_drop)
    hard_math_improved = (math_gain > 0.0) or (cmath_gain > 0.0)

    if hard_math_improved and retain_ok:
        recommendation = "advance_to_stage3"
    elif gsm_drop > 10.0 or ape_drop > 10.0:
        recommendation = "stop_and_mix_simple_data_10pct"
    elif gsm_drop > args.gsm8k_max_drop or ape_drop > args.ape210k_max_drop:
        recommendation = "reduce_lr_and_recheck"
    else:
        recommendation = "continue_stage2_for_next_epoch"

    out = {
        "stage1": {"gsm8k": gsm1, "ape210k": ape1, "math": math1, "cmath": cmath1},
        "stage2": {"gsm8k": gsm2, "ape210k": ape2, "math": math2, "cmath": cmath2},
        "delta": {"gsm8k_drop": gsm_drop, "ape210k_drop": ape_drop, "math_gain": math_gain, "cmath_gain": cmath_gain},
        "recommendation": recommendation,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
