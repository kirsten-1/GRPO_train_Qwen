#!/usr/bin/env python3
"""Offline evaluation for Stage2 (hard-math focus + retention checks)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

STAGE1_DIR = Path(__file__).resolve().parents[1] / "stage1"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from train_stage1_grpo import (
    DEFAULT_SYSTEM_PROMPT,
    InProcessGenerator,
    approx_word_count,
    parse_response,
    reasoning_quality_score,
    save_json,
    strip_code_fences,
)


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct"
DEFAULT_TRAINING_DATA_DIR = "/root/autodl-tmp/training_data"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/stage2_eval_runs"

DATASET_FILES = {
    "gsm8k": "gsm8k_test.json",
    "ape210k": "ape210k_test.json",
    "math": "math_test.json",
    "cmath": "cmath_test.json",
    "mbpp": "mbpp_test.json",
    "humaneval": "humaneval_test.json",
}

PRESET_SIZES = {
    "smoke": {"math": 128, "cmath": 128, "gsm8k": 128, "ape210k": 128, "mbpp": 32, "humaneval": 32},
    "core": {"math": None, "cmath": None, "gsm8k": 0, "ape210k": 0, "mbpp": 0, "humaneval": 0},
    "retain": {"math": 0, "cmath": 0, "gsm8k": None, "ape210k": None, "mbpp": 0, "humaneval": 0},
    "full": {"math": None, "cmath": None, "gsm8k": None, "ape210k": None, "mbpp": None, "humaneval": None},
    # Same math sample sizes as stage1/eval_stage1_model.py --suite baseline
    "stage1_math_aligned": {"math": 1187, "cmath": 1098, "gsm8k": 1319, "ape210k": 2000, "mbpp": 0, "humaneval": 0},
    # Same code sample sizes as baseline/code_math_baseline_data_summary.md
    "code_baseline_aligned": {"math": 0, "cmath": 0, "gsm8k": 0, "ape210k": 0, "mbpp": 257, "humaneval": 164},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage2 checkpoint/model.")
    p.add_argument("--model-path", required=True, help="Merged model or adapter/checkpoint dir.")
    p.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL, help="Needed when model-path is adapter-only.")
    p.add_argument("--training-data-dir", default=DEFAULT_TRAINING_DATA_DIR)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", default="stage2_eval")
    p.add_argument(
        "--suite",
        choices=["smoke", "core", "retain", "full", "stage1_math_aligned", "code_baseline_aligned"],
        default="full",
    )

    p.add_argument("--math-samples", type=int, default=None)
    p.add_argument("--cmath-samples", type=int, default=None)
    p.add_argument("--gsm8k-samples", type=int, default=None)
    p.add_argument("--ape210k-samples", type=int, default=None)
    p.add_argument("--mbpp-samples", type=int, default=None)
    p.add_argument("--humaneval-samples", type=int, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--math-max-new-tokens", type=int, default=512)
    p.add_argument("--code-max-new-tokens", type=int, default=1024)
    p.add_argument("--code-num-samples", type=int, default=5, help="Number of sampled candidates per code problem.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--timeout-s", type=float, default=6.0)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def has_full_model_weights(model_path: Path) -> bool:
    if (model_path / "pytorch_model.bin").exists() or (model_path / "model.safetensors").exists():
        return True
    for p in model_path.glob("model-*-of-*.safetensors"):
        if p.is_file():
            return True
    return False


def _print_progress(tag: str, done: int, total: int, start_ts: float, every: int) -> None:
    every = max(1, every)
    if done % every != 0 and done != total:
        return
    elapsed = time.time() - start_ts
    pct = (100.0 * done / total) if total else 100.0
    print(f"[{tag}] {done}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s", flush=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class CodeEvalResult:
    compile_ok: bool
    passed: bool
    reason: str
    runtime_s: float
    details: str = ""


def run_python_code(code: str, stdin_text: str = "", timeout_s: float = 6.0) -> Tuple[bool, subprocess.CompletedProcess[str], float, str]:
    code = strip_code_fences(code)
    with tempfile.TemporaryDirectory(prefix="stage2_eval_") as td:
        script_path = Path(td) / "main.py"
        script_path.write_text(code, encoding="utf-8")

        compile_proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script_path)],
            text=True,
            capture_output=True,
        )
        if compile_proc.returncode != 0:
            err = (compile_proc.stderr or "").strip()
            detail = f"compile_error: {err[:300]}" if err else "compile_error"
            dummy = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=compile_proc.stderr)
            return False, dummy, 0.0, detail

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                input=stdin_text,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
            elapsed = time.perf_counter() - start
            if proc.returncode != 0:
                return True, proc, elapsed, "runtime_error"
            return True, proc, elapsed, "ok"
        except subprocess.TimeoutExpired as e:
            elapsed = time.perf_counter() - start
            proc = subprocess.CompletedProcess(args=[], returncode=1, stdout=e.stdout or "", stderr=e.stderr or "")
            return True, proc, elapsed, "timeout"


def eval_mbpp_code(code: str, tests: List[str], timeout_s: float) -> CodeEvalResult:
    if not tests:
        return CodeEvalResult(compile_ok=False, passed=False, reason="missing_tests", runtime_s=0.0)
    wrapped = f"{strip_code_fences(code)}\n\n" + "\n".join(tests) + "\n"
    compile_ok, proc, elapsed, reason = run_python_code(wrapped, timeout_s=timeout_s)
    passed = compile_ok and proc.returncode == 0 and reason == "ok"
    details = proc.stderr[-600:] if proc and proc.stderr else ""
    return CodeEvalResult(
        compile_ok=compile_ok,
        passed=passed,
        reason=("pass" if passed else reason),
        runtime_s=elapsed,
        details=details,
    )


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


def extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


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
                "difficulty": str(item.get("difficulty", "unknown")),
            }
        )
    return maybe_sample(rows, max_samples, seed)


def parse_tests_from_prompt(user_text: str) -> List[str]:
    marker = "测试用例："
    if marker not in user_text:
        return []
    block = user_text.split(marker, 1)[1]
    return [line.strip() for line in block.splitlines() if line.strip()]


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


def parse_humaneval_from_user(user_text: str) -> Tuple[str, str]:
    marker = "测试用例："
    if marker in user_text:
        prompt, test = user_text.split(marker, 1)
        return prompt.strip(), test.strip()
    return user_text.strip(), ""


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
        rows.append(
            {
                "id": item.get("id", f"humaneval_{i}"),
                "system": system_prompt,
                "user": user_prompt,
                "difficulty": str(item.get("difficulty", "unknown")),
            }
        )
    return maybe_sample(rows, max_samples, seed)


def infer_function_name_from_prompt(prompt: str) -> Optional[str]:
    m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", prompt)
    return m.group(1) if m else None


def eval_humaneval_code(code: str, prompt: str, test_code: str, timeout_s: float) -> CodeEvalResult:
    fn_name = infer_function_name_from_prompt(prompt)
    if not fn_name:
        return CodeEvalResult(compile_ok=False, passed=False, reason="missing_function_name", runtime_s=0.0)

    wrapped = (
        f"{strip_code_fences(code)}\n\n"
        f"{test_code}\n\n"
        f"if __name__ == '__main__':\n"
        f"    check({fn_name})\n"
        f"    print('PASS')\n"
    )
    compile_ok, proc, elapsed, reason = run_python_code(wrapped, timeout_s=timeout_s)
    passed = compile_ok and proc.returncode == 0 and reason == "ok"
    details = proc.stderr[-600:] if proc and proc.stderr else ""
    return CodeEvalResult(
        compile_ok=compile_ok,
        passed=passed,
        reason=("pass" if passed else reason),
        runtime_s=elapsed,
        details=details,
    )


def evaluate_math(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: List[Dict[str, Any]],
    max_new_tokens: int,
    dataset_name: str,
    progress_every: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    gen = InProcessGenerator(model, tokenizer)
    total = len(rows)
    correct = 0
    verifiable = 0
    format_ok = 0
    reasoning_q_sum = 0.0
    reasoning_len_sum = 0
    by_diff: Dict[str, List[int]] = {}
    sample_records: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, s in enumerate(rows):
        out = gen.generate(
            system_prompt=s["system"],
            user_prompt=s["user"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )[0]
        parsed = parse_response(out)
        reward_value, is_verifiable = score_math_with_openr1(out, s["gold_answer"])
        ok = reward_value >= 0.5
        if ok:
            correct += 1
        if is_verifiable:
            verifiable += 1
        if parsed["format_ok"]:
            format_ok += 1
        reasoning_q = reasoning_quality_score(parsed["reasoning"])
        reasoning_len = approx_word_count(parsed["reasoning"])
        reasoning_q_sum += reasoning_q
        reasoning_len_sum += reasoning_len

        diff = s.get("difficulty", "unknown")
        if diff not in by_diff:
            by_diff[diff] = [0, 0]
        by_diff[diff][0] += 1
        if ok:
            by_diff[diff][1] += 1
        sample_records.append(
            {
                "id": s["id"],
                "dataset": dataset_name,
                "difficulty": diff,
                "question": s["user"],
                "gold_answer": s["gold_answer"],
                "pred_answer": parsed["answer"],
                "response": out,
                "accuracy_reward": reward_value,
                "is_correct": bool(ok),
                "is_verifiable": bool(is_verifiable),
                "format_ok": bool(parsed["format_ok"]),
                "reasoning_quality": reasoning_q,
                "reasoning_len_words": reasoning_len,
            }
        )
        _print_progress(dataset_name, i + 1, total, t0, progress_every)

    elapsed = time.time() - t0
    m: Dict[str, float] = {
        "accuracy": (correct / total) if total else 0.0,
        "accuracy_on_verifiable": (correct / verifiable) if verifiable else 0.0,
        "verifiable_rate": (verifiable / total) if total else 0.0,
        "num_verifiable_samples": float(verifiable),
        "format_correct_rate": (format_ok / total) if total else 0.0,
        "avg_reasoning_quality": (reasoning_q_sum / total) if total else 0.0,
        "avg_reasoning_length_words": (reasoning_len_sum / total) if total else 0.0,
        "num_samples": float(total),
        "elapsed_s": elapsed,
        "samples_per_second": (total / elapsed) if elapsed > 0 else 0.0,
    }
    for diff, (n, c) in sorted(by_diff.items()):
        key = re.sub(r"[^0-9A-Za-z_]+", "_", str(diff).strip().lower()) or "unknown"
        m[f"accuracy_by_difficulty__{key}"] = (c / n) if n else 0.0
        m[f"num_samples_by_difficulty__{key}"] = float(n)
    return m, sample_records


def evaluate_code_dataset(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: List[Dict[str, Any]],
    dataset_name: str,
    max_new_tokens: int,
    code_num_samples: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
    progress_every: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if dataset_name not in {"mbpp", "humaneval"}:
        raise ValueError(f"Unsupported code dataset: {dataset_name}")

    gen = InProcessGenerator(model, tokenizer)
    total = len(rows)
    pass1 = 0
    pass5 = 0
    pass10 = 0
    format_ok_cnt = 0
    compile_ok_total = 0
    passed_total = 0
    candidate_total = 0
    code_line_sum = 0
    runtime_sum = 0.0
    runtime_cnt = 0
    sample_records: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, s in enumerate(rows):
        outs = gen.generate(
            system_prompt=s["system"],
            user_prompt=s["user"],
            max_new_tokens=max_new_tokens,
            num_return_sequences=code_num_samples,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        results: List[CodeEvalResult] = []
        candidate_records: List[Dict[str, Any]] = []
        for j, out in enumerate(outs):
            parsed = parse_response(out)
            if parsed["format_ok"]:
                format_ok_cnt += 1
            code = strip_code_fences(parsed["answer"])
            code_line_sum += len([ln for ln in code.splitlines() if ln.strip()])

            if dataset_name == "mbpp":
                tests = parse_tests_from_prompt(s["user"])
                r = eval_mbpp_code(code, tests, timeout_s=timeout_s)
            else:
                prompt, test_code = parse_humaneval_from_user(s["user"])
                r = eval_humaneval_code(code, prompt=prompt, test_code=test_code, timeout_s=timeout_s)

            results.append(r)
            compile_ok_total += 1 if r.compile_ok else 0
            passed_total += 1 if r.passed else 0
            candidate_total += 1
            if r.runtime_s > 0:
                runtime_sum += r.runtime_s
                runtime_cnt += 1

            candidate_records.append(
                {
                    "candidate_index": j,
                    "passed": bool(r.passed),
                    "reason": r.reason,
                    "compile_ok": bool(r.compile_ok),
                    "pred_answer": parsed["answer"],
                    "response": out,
                }
            )

        top1 = results[:1]
        top5 = results[: min(5, len(results))]
        top10 = results[: min(10, len(results))]
        ok1 = any(r.passed for r in top1)
        ok5 = any(r.passed for r in top5)
        ok10 = any(r.passed for r in top10)
        pass1 += 1 if ok1 else 0
        pass5 += 1 if ok5 else 0
        pass10 += 1 if ok10 else 0

        sample_records.append(
            {
                "id": s["id"],
                "dataset": dataset_name,
                "difficulty": s.get("difficulty", "unknown"),
                "question": s["user"],
                "pass@1": bool(ok1),
                "pass@5": bool(ok5),
                "pass@10": bool(ok10),
                "candidates": candidate_records,
            }
        )
        _print_progress(dataset_name, i + 1, total, t0, progress_every)

    elapsed = time.time() - t0
    metrics = {
        "num_samples": float(total),
        "num_generations_per_sample": int(code_num_samples),
        "pass@1": (pass1 / total) if total else 0.0,
        "pass@5": (pass5 / total) if total else 0.0,
        "pass@10": (pass10 / total) if total else 0.0,
        "pass1": (pass1 / total) if total else 0.0,
        "compile_pass_rate": (compile_ok_total / candidate_total) if candidate_total else 0.0,
        "test_pass_rate": (passed_total / candidate_total) if candidate_total else 0.0,
        "format_correct_rate": (format_ok_cnt / candidate_total) if candidate_total else 0.0,
        "avg_code_lines": (code_line_sum / candidate_total) if candidate_total else 0.0,
        "avg_runtime_s": (runtime_sum / runtime_cnt) if runtime_cnt else 0.0,
        "elapsed_s": elapsed,
        "samples_per_second": (total / elapsed) if elapsed > 0 else 0.0,
    }
    return metrics, sample_records


def resolve_sizes(args: argparse.Namespace) -> Dict[str, Optional[int]]:
    sizes = dict(PRESET_SIZES[args.suite])
    overrides = {
        "math": args.math_samples,
        "cmath": args.cmath_samples,
        "gsm8k": args.gsm8k_samples,
        "ape210k": args.ape210k_samples,
        "mbpp": args.mbpp_samples,
        "humaneval": args.humaneval_samples,
    }
    for k, v in overrides.items():
        if v is not None:
            sizes[k] = v
    return sizes


def load_model_and_tokenizer(
    model_path: Path,
    base_model_path: Path,
    local_files_only: bool,
) -> Tuple[torch.nn.Module, Any, str]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter_only = (model_path / "adapter_config.json").exists() and not has_full_model_weights(model_path)
    if adapter_only:
        print(f"Adapter-only path detected, loading base model: {base_model_path}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(model, str(model_path), local_files_only=local_files_only)
        load_mode = "adapter_on_base"
        tokenizer_source = model_path if (model_path / "tokenizer_config.json").exists() else base_model_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        load_mode = "full_model"
        tokenizer_source = model_path

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = model.to(device)
    model.eval()
    return model, tokenizer, load_mode


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    base_model_path = Path(args.base_model_path)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{args.run_name}_{args.suite}_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sizes = resolve_sizes(args)
    if args.dry_run:
        print("DRY RUN")
        print("model_path:", model_path)
        print("base_model_path:", base_model_path)
        print("suite:", args.suite)
        print("sizes:", sizes)
        print("output:", run_dir)
        return

    data_dir = Path(args.training_data_dir)
    files = {k: data_dir / v for k, v in DATASET_FILES.items()}
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file for {k}: {p}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model path: {model_path}")
    if not base_model_path.exists():
        raise FileNotFoundError(f"Missing base model path: {base_model_path}")

    model, tokenizer, load_mode = load_model_and_tokenizer(
        model_path=model_path,
        base_model_path=base_model_path,
        local_files_only=args.local_files_only,
    )

    suites: Dict[str, List[Dict[str, Any]]] = {
        # Keep seed offsets aligned with stage1/eval_stage1_model.py for strict comparability.
        "gsm8k": load_math_samples(files["gsm8k"], sizes["gsm8k"], args.seed + 11),
        "ape210k": load_math_samples(files["ape210k"], sizes["ape210k"], args.seed + 12),
        "math": load_math_samples(files["math"], sizes["math"], args.seed + 13),
        "cmath": load_math_samples(files["cmath"], sizes["cmath"], args.seed + 14),
        "mbpp": load_mbpp_samples(files["mbpp"], sizes["mbpp"], args.seed + 15),
        "humaneval": load_humaneval_samples(files["humaneval"], sizes["humaneval"], args.seed + 16),
    }

    metrics: Dict[str, float] = {}
    sample_records: List[Dict[str, Any]] = []
    for name in ("math", "cmath", "gsm8k", "ape210k"):
        rows = suites[name]
        if not rows:
            continue
        print(f"[{name}] start: {len(rows)}", flush=True)
        m, recs = evaluate_math(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            max_new_tokens=args.math_max_new_tokens,
            dataset_name=name,
            progress_every=args.progress_every,
        )
        for k, v in m.items():
            metrics[f"{name}_{k}"] = v
        sample_records.extend(recs)
    for name in ("mbpp", "humaneval"):
        rows = suites[name]
        if not rows:
            continue
        print(f"[{name}] start: {len(rows)}", flush=True)
        m, recs = evaluate_code_dataset(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            dataset_name=name,
            max_new_tokens=args.code_max_new_tokens,
            code_num_samples=args.code_num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout_s=args.timeout_s,
            progress_every=args.progress_every,
        )
        for k, v in m.items():
            metrics[f"{name}_{k}"] = v
        sample_records.extend(recs)

    samples_jsonl = run_dir / "sample_generations.jsonl"
    write_jsonl(sample_records, samples_jsonl)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "base_model_path": str(base_model_path),
        "load_mode": load_mode,
        "suite": args.suite,
        "sample_sizes": sizes,
        "metrics": metrics,
        "num_recorded_samples": len(sample_records),
        "sample_generations_file": str(samples_jsonl),
        "sample_generations": sample_records,
    }
    save_json(payload, run_dir / "eval_results.json")
    print("Evaluation complete")
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()
