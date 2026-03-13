#!/usr/bin/env python3
"""Offline evaluation for Stage1 models on math and optional code benchmarks."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
OPENR1_SRC_DIR = ROOT_DIR / "open-r1" / "src"
if str(OPENR1_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(OPENR1_SRC_DIR))

from train_stage1_grpo import (
    DEFAULT_SYSTEM_PROMPT,
    InProcessGenerator,
    approx_word_count,
    eval_mbpp_code,
    parse_response,
    reasoning_quality_score,
    run_python_code,
    save_json,
    strip_code_fences,
)


DEFAULT_TRAINING_DATA_DIR = "/root/autodl-tmp/training_data"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/stage1_eval_runs"

DATASET_FILES = {
    "gsm8k": "gsm8k_test.json",
    "ape210k": "ape210k_test.json",
    "math": "math_test.json",
    "cmath": "cmath_test.json",
    "mbpp": "mbpp_test.json",
    "humaneval": "humaneval_test.json",
    "apps": "apps_test.json",
}

# Baseline-aligned targets for Stage1 report.
PRESET_SIZES = {
    "smoke": {
        "gsm8k": 64,
        "ape210k": 128,
        "math": 64,
        "cmath": 64,
        "mbpp": 32,
        "humaneval": 32,
        "apps": 64,
    },
    "baseline": {
        "gsm8k": 1319,
        "ape210k": 2000,
        "math": 1187,
        "cmath": 1098,
        "mbpp": 257,
        "humaneval": 164,
        "apps": 500,
    },
    "full": {
        "gsm8k": None,
        "ape210k": None,
        "math": None,
        "cmath": None,
        "mbpp": None,
        "humaneval": None,
        "apps": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate merged Stage1 model.")
    parser.add_argument("--model-path", required=True, help="Merged model path.")
    parser.add_argument("--training-data-dir", default=DEFAULT_TRAINING_DATA_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="stage1_eval")
    parser.add_argument("--suite", choices=["smoke", "baseline", "full"], default="baseline")
    parser.add_argument(
        "--code-benchmarks",
        choices=["none", "mbpp", "mbpp_humaneval", "all"],
        default="none",
        help="Optional code evaluation set.",
    )

    parser.add_argument("--gsm8k-samples", type=int, default=None)
    parser.add_argument("--ape210k-samples", type=int, default=None)
    parser.add_argument("--math-samples", type=int, default=None)
    parser.add_argument("--cmath-samples", type=int, default=None)
    parser.add_argument("--mbpp-samples", type=int, default=None)
    parser.add_argument("--humaneval-samples", type=int, default=None)
    parser.add_argument("--apps-samples", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--math-max-new-tokens", type=int, default=512)
    parser.add_argument("--code-max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N samples.")
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tqdm progress bars. Disable with --no-progress-bar.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_math_with_openr1(full_completion: str, gold_answer: str) -> Tuple[float, bool]:
    gold_parsed = parse(str(gold_answer), extraction_mode="first_match")
    if len(gold_parsed) == 0:
        return 0.0, False
    answer_parsed = parse(
        full_completion,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    try:
        return float(verify(gold_parsed, answer_parsed)), True
    except Exception:
        return 0.0, False


def maybe_sample(items: List[Dict[str, Any]], max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_samples is None:
        return items
    if max_samples == 0:
        return []
    if max_samples < 0 or len(items) <= max_samples:
        return items
    rng = random.Random(seed)
    return rng.sample(items, max_samples)


def load_math_samples(path: Path, max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    data = load_json(path)
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        msgs = item.get("messages", [])
        if len(msgs) < 3:
            continue
        system_prompt = str(msgs[0].get("content", "")).strip() or DEFAULT_SYSTEM_PROMPT
        user_prompt = str(msgs[1].get("content", "")).strip()
        gold_answer = extract_tag_block(str(msgs[2].get("content", "")), "answer").strip()
        if not user_prompt:
            continue
        rows.append(
            {
                "id": item.get("id", f"{path.stem}_{i}"),
                "system": system_prompt,
                "user": user_prompt,
                "gold_answer": gold_answer,
            }
        )
    return maybe_sample(rows, max_samples, seed)


def parse_tests_from_prompt(user_text: str) -> List[str]:
    marker = "测试用例："
    if marker not in user_text:
        return []
    block = user_text.split(marker, 1)[1]
    return [line.rstrip() for line in block.splitlines() if line.strip()]


def load_mbpp_samples(path: Path, max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    data = load_json(path)
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        msgs = item.get("messages", [])
        if len(msgs) < 2:
            continue
        system_prompt = str(msgs[0].get("content", "")).strip() or DEFAULT_SYSTEM_PROMPT
        user_prompt = str(msgs[1].get("content", "")).strip()
        if not user_prompt:
            continue
        rows.append(
            {
                "id": item.get("id", f"mbpp_{i}"),
                "system": system_prompt,
                "user": user_prompt,
                "tests": parse_tests_from_prompt(user_prompt),
            }
        )
    return maybe_sample(rows, max_samples, seed)


def load_humaneval_samples(path: Path, max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    data = load_json(path)
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        msgs = item.get("messages", [])
        if len(msgs) < 2:
            continue
        system_prompt = str(msgs[0].get("content", "")).strip() or DEFAULT_SYSTEM_PROMPT
        user_prompt = str(msgs[1].get("content", "")).strip()
        if not user_prompt:
            continue
        marker = "测试用例："
        tests_block = user_prompt.split(marker, 1)[1].strip() if marker in user_prompt else ""
        rows.append(
            {
                "id": item.get("id", f"humaneval_{i}"),
                "system": system_prompt,
                "user": user_prompt,
                "tests_block": tests_block,
            }
        )
    return maybe_sample(rows, max_samples, seed)


def load_apps_samples(path: Path, max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    data = load_json(path)
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        msgs = item.get("messages", [])
        if len(msgs) < 2:
            continue
        system_prompt = str(msgs[0].get("content", "")).strip() or DEFAULT_SYSTEM_PROMPT
        user_prompt = str(msgs[1].get("content", "")).strip()
        if not user_prompt:
            continue
        rows.append(
            {
                "id": item.get("id", f"apps_{i}"),
                "system": system_prompt,
                "user": user_prompt,
            }
        )
    return maybe_sample(rows, max_samples, seed)


def extract_first_function_name(code: str) -> Optional[str]:
    m = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code, flags=re.MULTILINE)
    return m.group(1) if m else None


def compile_only_python(code: str) -> bool:
    code = strip_code_fences(code)
    with tempfile.TemporaryDirectory(prefix="stage1_compile_") as td:
        p = Path(td) / "main.py"
        p.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            ["python", "-m", "py_compile", str(p)],
            text=True,
            capture_output=True,
        )
    return proc.returncode == 0


def evaluate_humaneval_proxy(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    max_new_tokens: int,
    timeout_s: float,
    progress_every: int,
    progress_bar: bool,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    gen = InProcessGenerator(model, tokenizer)
    total = len(samples)
    pass_cnt = 0
    compile_ok_cnt = 0
    has_func_cnt = 0
    testable_cnt = 0
    sample_generations: List[Dict[str, Any]] = []
    start_ts = time.time()

    iterator = tqdm(samples, desc="humaneval", dynamic_ncols=True) if progress_bar else samples
    for i, s in enumerate(iterator):
        out = gen.generate(
            system_prompt=s["system"],
            user_prompt=s["user"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )[0]
        parsed = parse_response(out)
        code = strip_code_fences(parsed["answer"])
        func_name = extract_first_function_name(code)

        reason = "missing_function"
        passed = False
        compile_ok = compile_only_python(code)
        if compile_ok:
            compile_ok_cnt += 1
        if func_name:
            has_func_cnt += 1
            if s["tests_block"]:
                testable_cnt += 1
                script = (
                    f"{code}\n\n{s['tests_block']}\n\n"
                    f"if __name__ == '__main__':\n    check({func_name})\n"
                )
                compile_ok2, proc, _elapsed, reason = run_python_code(script, timeout_s=timeout_s)
                if compile_ok2 and proc.returncode == 0 and reason == "ok":
                    passed = True
                    pass_cnt += 1
        sample_generations.append(
            {
                "id": s["id"],
                "dataset": "humaneval",
                "passed": bool(passed),
                "reason": reason,
                "compile_ok": bool(compile_ok),
                "pred_answer": parsed["answer"],
                "response": out,
            }
        )
        if not progress_bar:
            _print_progress("humaneval", i + 1, total, start_ts, progress_every)

    metrics = {
        "pass1_proxy": (pass_cnt / total) if total else 0.0,
        "compile_pass_rate": (compile_ok_cnt / total) if total else 0.0,
        "has_function_rate": (has_func_cnt / total) if total else 0.0,
        "testable_prompt_rate": (testable_cnt / total) if total else 0.0,
        "num_samples": float(total),
    }
    return metrics, sample_generations


def evaluate_apps_proxy(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    max_new_tokens: int,
    progress_every: int,
    progress_bar: bool,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    gen = InProcessGenerator(model, tokenizer)
    total = len(samples)
    code_like_cnt = 0
    compile_ok_cnt = 0
    sample_generations: List[Dict[str, Any]] = []
    code_like_pattern = re.compile(r"\b(def|class|import|for|while|if|return|print)\b")
    start_ts = time.time()

    iterator = tqdm(samples, desc="apps", dynamic_ncols=True) if progress_bar else samples
    for i, s in enumerate(iterator):
        out = gen.generate(
            system_prompt=s["system"],
            user_prompt=s["user"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )[0]
        parsed = parse_response(out)
        code = strip_code_fences(parsed["answer"])
        is_code_like = bool(code_like_pattern.search(code))
        if is_code_like:
            code_like_cnt += 1
        code_compile_ok = compile_only_python(code)
        if code_compile_ok:
            compile_ok_cnt += 1
        sample_generations.append(
            {
                "id": s["id"],
                "dataset": "apps",
                "code_like": bool(is_code_like),
                "compile_ok": bool(code_compile_ok),
                "pred_answer": parsed["answer"],
                "response": out,
            }
        )
        if not progress_bar:
            _print_progress("apps", i + 1, total, start_ts, progress_every)

    metrics = {
        "code_like_rate": (code_like_cnt / total) if total else 0.0,
        "compile_pass_rate": (compile_ok_cnt / total) if total else 0.0,
        "num_samples": float(total),
    }
    return metrics, sample_generations


def resolve_sample_sizes(args: argparse.Namespace) -> Dict[str, Optional[int]]:
    sizes = dict(PRESET_SIZES[args.suite])
    overrides = {
        "gsm8k": args.gsm8k_samples,
        "ape210k": args.ape210k_samples,
        "math": args.math_samples,
        "cmath": args.cmath_samples,
        "mbpp": args.mbpp_samples,
        "humaneval": args.humaneval_samples,
        "apps": args.apps_samples,
    }
    for key, value in overrides.items():
        if value is not None:
            sizes[key] = value
    return sizes


def _print_progress(tag: str, done: int, total: int, start_ts: float, progress_every: int) -> None:
    if progress_every <= 0:
        progress_every = 1
    if done % progress_every != 0 and done != total:
        return
    elapsed = time.time() - start_ts
    pct = (100.0 * done / total) if total else 100.0
    print(f"[{tag}] {done}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s", flush=True)


def evaluate_math_verbose(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    max_new_tokens: int,
    dataset_name: str,
    progress_every: int,
    progress_bar: bool,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    gen = InProcessGenerator(model, tokenizer)
    total = len(samples)
    correct = 0
    verifiable = 0
    format_ok = 0
    reasoning_q_sum = 0.0
    reasoning_len_sum = 0
    sample_generations: List[Dict[str, Any]] = []
    start_ts = time.time()

    iterator = tqdm(samples, desc=dataset_name, dynamic_ncols=True) if progress_bar else samples
    for i, s in enumerate(iterator):
        out = gen.generate(
            system_prompt=s["system"],
            user_prompt=s["user"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )[0]
        parsed = parse_response(out)
        reward_value, is_verifiable = score_math_with_openr1(out, s["gold_answer"])
        is_correct = reward_value >= 0.5
        if is_correct:
            correct += 1
        if is_verifiable:
            verifiable += 1
        if parsed["format_ok"]:
            format_ok += 1
        reasoning_q = reasoning_quality_score(parsed["reasoning"])
        reasoning_len = approx_word_count(parsed["reasoning"])
        reasoning_q_sum += reasoning_q
        reasoning_len_sum += reasoning_len

        sample_generations.append(
            {
                "id": s["id"],
                "dataset": dataset_name,
                "question": s["user"],
                "gold_answer": s["gold_answer"],
                "pred_answer": parsed["answer"],
                "response": out,
                "accuracy_reward": reward_value,
                "is_correct": bool(is_correct),
                "is_verifiable": bool(is_verifiable),
                "format_ok": bool(parsed["format_ok"]),
                "reasoning_quality": reasoning_q,
                "reasoning_len_words": reasoning_len,
            }
        )
        if not progress_bar:
            _print_progress(dataset_name, i + 1, total, start_ts, progress_every)

    metrics = {
        "accuracy": (correct / total) if total else 0.0,
        "accuracy_on_verifiable": (correct / verifiable) if verifiable else 0.0,
        "verifiable_rate": (verifiable / total) if total else 0.0,
        "num_verifiable_samples": float(verifiable),
        "format_correct_rate": (format_ok / total) if total else 0.0,
        "avg_reasoning_quality": (reasoning_q_sum / total) if total else 0.0,
        "avg_reasoning_length_words": (reasoning_len_sum / total) if total else 0.0,
        "num_samples": float(total),
    }
    return metrics, sample_generations


def evaluate_mbpp_verbose(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    max_new_tokens: int,
    timeout_s: float,
    progress_every: int,
    progress_bar: bool,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    gen = InProcessGenerator(model, tokenizer)
    total = len(samples)
    passed = 0
    compile_ok = 0
    sample_generations: List[Dict[str, Any]] = []
    start_ts = time.time()

    iterator = tqdm(samples, desc="mbpp", dynamic_ncols=True) if progress_bar else samples
    for i, s in enumerate(iterator):
        out = gen.generate(
            system_prompt=s["system"],
            user_prompt=s["user"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )[0]
        parsed = parse_response(out)
        code = strip_code_fences(parsed["answer"])
        r = eval_mbpp_code(code, s.get("tests", []), timeout_s=timeout_s)
        if r.compile_ok:
            compile_ok += 1
        if r.passed:
            passed += 1
        sample_generations.append(
            {
                "id": s["id"],
                "dataset": "mbpp",
                "question": s["user"],
                "passed": bool(r.passed),
                "reason": r.reason,
                "compile_ok": bool(r.compile_ok),
                "pred_answer": parsed["answer"],
                "response": out,
            }
        )
        if not progress_bar:
            _print_progress("mbpp", i + 1, total, start_ts, progress_every)

    metrics = {
        "pass1": (passed / total) if total else 0.0,
        "compile_pass_rate": (compile_ok / total) if total else 0.0,
        "num_samples": float(total),
    }
    return metrics, sample_generations


def enabled_code_sets(code_benchmarks: str) -> Dict[str, bool]:
    return {
        "mbpp": code_benchmarks in {"mbpp", "mbpp_humaneval", "all"},
        "humaneval": code_benchmarks in {"mbpp_humaneval", "all"},
        "apps": code_benchmarks == "all",
    }


def main() -> None:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{args.run_name}_{args.suite}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sizes = resolve_sample_sizes(args)
    code_flags = enabled_code_sets(args.code_benchmarks)

    if args.dry_run:
        print("DRY RUN")
        print("model_path:", args.model_path)
        print("suite:", args.suite)
        print("code_benchmarks:", args.code_benchmarks)
        print("progress_every:", args.progress_every)
        print("sample_sizes:", sizes)
        print("output_run_dir:", run_dir)
        return

    data_dir = Path(args.training_data_dir)
    file_map = {k: data_dir / v for k, v in DATASET_FILES.items()}
    for key, p in file_map.items():
        if key in {"mbpp", "humaneval", "apps"}:
            continue
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")
    if code_flags["mbpp"] and not file_map["mbpp"].exists():
        raise FileNotFoundError(f"Missing dataset file: {file_map['mbpp']}")
    if code_flags["humaneval"] and not file_map["humaneval"].exists():
        raise FileNotFoundError(f"Missing dataset file: {file_map['humaneval']}")
    if code_flags["apps"] and not file_map["apps"].exists():
        raise FileNotFoundError(f"Missing dataset file: {file_map['apps']}")

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "dtype": dtype,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("dtype", None)
        model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    suites: Dict[str, List[Dict[str, Any]]] = {
        "gsm8k": load_math_samples(file_map["gsm8k"], sizes["gsm8k"], args.seed + 11),
        "ape210k": load_math_samples(file_map["ape210k"], sizes["ape210k"], args.seed + 12),
        "math": load_math_samples(file_map["math"], sizes["math"], args.seed + 13),
        "cmath": load_math_samples(file_map["cmath"], sizes["cmath"], args.seed + 14),
    }
    if code_flags["mbpp"]:
        suites["mbpp"] = load_mbpp_samples(file_map["mbpp"], sizes["mbpp"], args.seed + 15)

    t0 = time.time()
    metrics: Dict[str, float] = {}
    sample_records: List[Dict[str, Any]] = []

    for name in ("gsm8k", "ape210k", "math", "cmath"):
        if name not in suites:
            continue
        rows = suites[name]
        print(f"[{name}] start: {len(rows)} samples", flush=True)
        m, gens = evaluate_math_verbose(
            model=model,
            tokenizer=tokenizer,
            samples=rows,
            max_new_tokens=args.math_max_new_tokens,
            dataset_name=name,
            progress_every=args.progress_every,
            progress_bar=args.progress_bar,
        )
        for k, v in m.items():
            metrics[f"{name}_{k}"] = v
        sample_records.extend(gens)

    if "mbpp" in suites:
        rows = suites["mbpp"]
        print(f"[mbpp] start: {len(rows)} samples", flush=True)
        m, gens = evaluate_mbpp_verbose(
            model=model,
            tokenizer=tokenizer,
            samples=rows,
            max_new_tokens=args.code_max_new_tokens,
            timeout_s=args.timeout_s,
            progress_every=args.progress_every,
            progress_bar=args.progress_bar,
        )
        for k, v in m.items():
            metrics[f"mbpp_{k}"] = v
        sample_records.extend(gens)

    if "gsm8k_avg_reasoning_quality" in metrics and "ape210k_avg_reasoning_quality" in metrics:
        metrics["avg_reasoning_quality"] = (
            metrics["gsm8k_avg_reasoning_quality"] + metrics["ape210k_avg_reasoning_quality"]
        ) / 2.0
    if "gsm8k_avg_reasoning_length_words" in metrics and "ape210k_avg_reasoning_length_words" in metrics:
        metrics["avg_reasoning_length_words"] = (
            metrics["gsm8k_avg_reasoning_length_words"] + metrics["ape210k_avg_reasoning_length_words"]
        ) / 2.0

    elapsed_core = time.time() - t0

    if code_flags["humaneval"]:
        humaneval_rows = load_humaneval_samples(file_map["humaneval"], sizes["humaneval"], args.seed + 21)
        t1 = time.time()
        hm_metrics, hm_preview = evaluate_humaneval_proxy(
            model=model,
            tokenizer=tokenizer,
            samples=humaneval_rows,
            max_new_tokens=args.code_max_new_tokens,
            timeout_s=args.timeout_s,
            progress_every=args.progress_every,
            progress_bar=args.progress_bar,
        )
        metrics.update({f"humaneval_{k}": v for k, v in hm_metrics.items()})
        metrics["humaneval_elapsed_s"] = time.time() - t1
        sample_records.extend(hm_preview)

    if code_flags["apps"]:
        apps_rows = load_apps_samples(file_map["apps"], sizes["apps"], args.seed + 31)
        t2 = time.time()
        apps_metrics, apps_preview = evaluate_apps_proxy(
            model=model,
            tokenizer=tokenizer,
            samples=apps_rows,
            max_new_tokens=args.code_max_new_tokens,
            progress_every=args.progress_every,
            progress_bar=args.progress_bar,
        )
        metrics.update({f"apps_{k}": v for k, v in apps_metrics.items()})
        metrics["apps_elapsed_s"] = time.time() - t2
        sample_records.extend(apps_preview)

    samples_jsonl = run_dir / "sample_generations.jsonl"
    write_jsonl(sample_records, samples_jsonl)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "suite": args.suite,
        "code_benchmarks": args.code_benchmarks,
        "sample_sizes": sizes,
        "metrics": metrics,
        "elapsed_core_s": elapsed_core,
        "num_recorded_samples": len(sample_records),
        "sample_generations_file": str(samples_jsonl),
        "sample_generations": sample_records,
    }
    save_json(payload, run_dir / "eval_results.json")

    summary = {k: v for k, v in metrics.items() if k.endswith("accuracy") or k.endswith("pass1") or "num_samples" in k}
    print("Evaluation complete")
    print("Run dir:", run_dir)
    print("Summary metrics:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
