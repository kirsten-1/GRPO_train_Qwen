#!/usr/bin/env python3
"""
Stage1 GRPO training entrypoint for simple math (GSM8K + Ape210K).

Features:
- Optional Ape210K downsampling to enforce Ape:GSM ratio (default 3:1)
- LoRA GRPO training
- Quick eval every N steps
- Epoch eval + early stopping
- Final eval + success/warning criteria report
- Best-checkpoint saving and metric logging
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback


DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct"
DEFAULT_STAGE1_TRAIN_PATH = "/root/autodl-tmp/training_data/stage1_simple_math_train_ratio3.json"
DEFAULT_TRAINING_DATA_DIR = "/root/autodl-tmp/training_data"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/stage1_grpo_runs"
STAGE1_REWARD_WEIGHTS = (1.0, 0.2, 0.1)

DEFAULT_SYSTEM_PROMPT = """你是一个专业的AI助手，擅长数学推理和编程。请一步步思考并解决问题。

回答格式：

<reasoning>
在这里展示你的思考过程：
- 理解问题的关键点
- 分析解题思路
- 展示详细的推导或实现步骤
- 验证答案的正确性
</reasoning>

<answer>
在这里给出最终答案（数学题给出数值，编程题给出完整代码）
</answer>

注意：推理过程要详细、有逻辑，但避免冗余重复。"""

BASELINE_METRICS = {
    "gsm8k_accuracy": 22.52,
    "ape210k_accuracy": 11.65,
    "gsm8k_reasoning_quality": 0.783,
    "ape210k_reasoning_quality": 0.670,
    "avg_reasoning_quality": 0.727,
    "gsm8k_format_correct": 0.95,
    "ape210k_format_correct": 0.93,
    "math_accuracy": 0.00,
    "cmath_accuracy": 11.66,
    "mbpp_pass1": 57.2,
}


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    sample_generations: List[Dict[str, str]]


@dataclass
class CodeEvalResult:
    compile_ok: bool
    passed: bool
    reason: str
    runtime_s: float
    details: str = ""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_response(text: str) -> Dict[str, Any]:
    reasoning = extract_tag_block(text, "reasoning")
    answer = extract_tag_block(text, "answer")
    format_ok = bool(reasoning and answer)
    if not answer:
        answer = text.strip()
    return {"reasoning": reasoning, "answer": answer, "format_ok": format_ok}


def strip_code_fences(code: str) -> str:
    code = code.strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else code


def approx_word_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text))


def reasoning_quality_score(reasoning: str) -> float:
    if not reasoning:
        return 0.0
    wc = approx_word_count(reasoning)
    length_score = min(wc / 120.0, 1.0)
    has_steps = 1.0 if re.search(r"(1[\.\、]|step|first|then|finally|思路|过程|步骤|首先|然后|最后|解题思路)", reasoning, flags=re.IGNORECASE) else 0.0
    has_verify = 1.0 if re.search(r"(验证|check|verify|test|assert|合理|correct|检查|合理性)", reasoning, flags=re.IGNORECASE) else 0.0
    return 0.45 * length_score + 0.3 * has_steps + 0.25 * has_verify


def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("，", ",").replace("。", ".").replace("：", ":").replace("％", "%")
    s = re.sub(r"\s+", "", s)
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


def safe_eval_math_expr(expr: str) -> Optional[float]:
    expr = expr.strip()
    if not expr:
        return None

    mm = re.fullmatch(r"(-?\d+)\((\d+)/(\d+)\)", expr)
    if mm:
        whole = float(mm.group(1))
        num = float(mm.group(2))
        den = float(mm.group(3))
        if den == 0:
            return None
        sign = -1.0 if whole < 0 else 1.0
        return whole + sign * (num / den)

    if expr.endswith("%"):
        base = safe_eval_math_expr(expr[:-1])
        return None if base is None else base / 100.0

    expr = expr.replace("^", "**")
    expr = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: f"({m.group(1)}/100)", expr)

    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Mod,
        ast.FloorDiv,
        ast.Load,
    )
    try:
        node = ast.parse(expr, mode="eval")
        for n in ast.walk(node):
            if not isinstance(n, allowed):
                return None
        val = eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, {})
        if isinstance(val, (int, float)):
            return float(val)
    except Exception:
        return None
    return None


def math_answers_equal(pred: str, gold: str, tol: float = 1e-6) -> bool:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if p == g:
        return True
    pv = safe_eval_math_expr(p)
    gv = safe_eval_math_expr(g)
    if pv is not None and gv is not None:
        return math.isclose(pv, gv, rel_tol=tol, abs_tol=tol)
    return False


def parse_mbpp_tests_from_user(user_text: str) -> List[str]:
    marker = "测试用例："
    if marker not in user_text:
        return []
    block = user_text.split(marker, 1)[1]
    return [line.strip() for line in block.splitlines() if line.strip()]


def run_python_code(code: str, timeout_s: float = 6.0) -> Tuple[bool, subprocess.CompletedProcess[str], float, str]:
    code = strip_code_fences(code)
    with tempfile.TemporaryDirectory(prefix="stage1_eval_") as td:
        script_path = Path(td) / "main.py"
        script_path.write_text(code, encoding="utf-8")

        compile_proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script_path)],
            text=True,
            capture_output=True,
        )
        if compile_proc.returncode != 0:
            dummy = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=compile_proc.stderr)
            return False, dummy, 0.0, "compile_error"

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
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
    details = proc.stderr[-400:] if proc and proc.stderr else ""
    return CodeEvalResult(compile_ok=compile_ok, passed=passed, reason=("pass" if passed else reason), runtime_s=elapsed, details=details)


def rebalance_stage1(samples: List[Dict[str, Any]], ape_to_gsm_ratio: float, seed: int) -> List[Dict[str, Any]]:
    gsm = [x for x in samples if x.get("dataset") == "gsm8k"]
    ape = [x for x in samples if x.get("dataset") == "ape210k"]
    if not gsm or not ape:
        return samples
    rng = random.Random(seed)
    target_ape = int(round(len(gsm) * ape_to_gsm_ratio))
    if target_ape <= 0:
        raise ValueError(f"Invalid target Ape sample size {target_ape}.")
    if len(ape) > target_ape:
        ape = rng.sample(ape, target_ape)
    mixed = gsm + ape
    rng.shuffle(mixed)
    return mixed


def stage1_to_train_records(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, item in enumerate(samples):
        messages = item.get("messages", [])
        if len(messages) < 3:
            continue
        user_prompt = str(messages[1].get("content", "")).strip()
        gold_assistant = str(messages[2].get("content", ""))
        gold_answer = extract_tag_block(gold_assistant, "answer")
        if not gold_answer:
            gold_answer = gold_assistant.strip()
        if not user_prompt:
            continue
        records.append(
            {
                "id": item.get("id", f"stage1_{i}"),
                "dataset": item.get("dataset", "unknown"),
                "question": user_prompt,
                "solution": gold_answer,
            }
        )
    return records


def load_math_eval_samples(path: Path, max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    data = load_json(path)
    parsed: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        msgs = item.get("messages", [])
        if len(msgs) < 3:
            continue
        system_prompt = str(msgs[0].get("content", "")).strip()
        user_prompt = str(msgs[1].get("content", "")).strip()
        gold_answer = extract_tag_block(str(msgs[2].get("content", "")), "answer")
        if not user_prompt:
            continue
        parsed.append(
            {
                "id": item.get("id", f"s_{i}"),
                "system": system_prompt or DEFAULT_SYSTEM_PROMPT,
                "user": user_prompt,
                "gold_answer": gold_answer.strip(),
            }
        )
    if max_samples is not None and max_samples > 0 and len(parsed) > max_samples:
        rng = random.Random(seed)
        parsed = rng.sample(parsed, max_samples)
    return parsed


def load_mbpp_eval_samples(path: Path, max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    data = load_json(path)
    parsed: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        msgs = item.get("messages", [])
        if len(msgs) < 2:
            continue
        system_prompt = str(msgs[0].get("content", "")).strip() if len(msgs) >= 1 else DEFAULT_SYSTEM_PROMPT
        user_prompt = str(msgs[1].get("content", "")).strip()
        tests = parse_mbpp_tests_from_user(user_prompt)
        if not user_prompt:
            continue
        parsed.append(
            {
                "id": item.get("id", f"mbpp_{i}"),
                "system": system_prompt or DEFAULT_SYSTEM_PROMPT,
                "user": user_prompt,
                "tests": tests,
            }
        )
    if max_samples is not None and max_samples > 0 and len(parsed) > max_samples:
        rng = random.Random(seed)
        parsed = rng.sample(parsed, max_samples)
    return parsed


class InProcessGenerator:
    def __init__(self, model: torch.nn.Module, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.input_device = next(self.model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> List[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if isinstance(prompt_inputs, torch.Tensor):
            prompt_ids = prompt_inputs
            attention_mask = torch.ones_like(prompt_ids)
        elif isinstance(prompt_inputs, Mapping):
            prompt_ids = prompt_inputs.get("input_ids")
            attention_mask = prompt_inputs.get("attention_mask")
            if prompt_ids is None:
                raise ValueError("Missing input_ids from chat template output.")
            if attention_mask is None:
                attention_mask = torch.ones_like(prompt_ids)
        else:
            raise TypeError(f"Unsupported chat template output: {type(prompt_inputs)}")

        prompt_ids = prompt_ids.to(self.input_device)
        attention_mask = attention_mask.to(self.input_device)

        if not do_sample:
            temperature = 1.0

        outputs = self.model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = prompt_ids.shape[-1]
        texts: List[str] = []
        for seq in outputs:
            texts.append(self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
        return texts


class OnlineEvaluator:
    def __init__(self, model: torch.nn.Module, tokenizer: Any, timeout_s: float = 6.0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.timeout_s = timeout_s

    def evaluate_math(self, samples: List[Dict[str, Any]], max_new_tokens: int) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
        gen = InProcessGenerator(self.model, self.tokenizer)
        total = len(samples)
        correct = 0
        format_ok = 0
        reasoning_q_sum = 0.0
        reasoning_len_sum = 0
        sample_generations: List[Dict[str, str]] = []

        for i, s in enumerate(samples):
            out = gen.generate(
                system_prompt=s["system"],
                user_prompt=s["user"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )[0]
            parsed = parse_response(out)
            is_correct = math_answers_equal(parsed["answer"], s["gold_answer"])
            if is_correct:
                correct += 1
            if parsed["format_ok"]:
                format_ok += 1
            reasoning_q_sum += reasoning_quality_score(parsed["reasoning"])
            reasoning_len_sum += approx_word_count(parsed["reasoning"])

            if i < 3:
                sample_generations.append(
                    {
                        "id": s["id"],
                        "question": s["user"][:240],
                        "gold_answer": s["gold_answer"],
                        "pred_answer": parsed["answer"][:180],
                        "response_preview": out[:360],
                    }
                )

        metrics = {
            "accuracy": (correct / total) if total else 0.0,
            "format_correct_rate": (format_ok / total) if total else 0.0,
            "avg_reasoning_quality": (reasoning_q_sum / total) if total else 0.0,
            "avg_reasoning_length_words": (reasoning_len_sum / total) if total else 0.0,
            "num_samples": float(total),
        }
        return metrics, sample_generations

    def evaluate_mbpp(self, samples: List[Dict[str, Any]], max_new_tokens: int) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
        gen = InProcessGenerator(self.model, self.tokenizer)
        total = len(samples)
        passed = 0
        compile_ok = 0
        sample_generations: List[Dict[str, str]] = []

        for i, s in enumerate(samples):
            out = gen.generate(
                system_prompt=s["system"],
                user_prompt=s["user"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )[0]
            parsed = parse_response(out)
            code = strip_code_fences(parsed["answer"])
            r = eval_mbpp_code(code, s.get("tests", []), timeout_s=self.timeout_s)
            if r.compile_ok:
                compile_ok += 1
            if r.passed:
                passed += 1
            if i < 3:
                sample_generations.append(
                    {
                        "id": s["id"],
                        "question": s["user"][:200],
                        "passed": str(r.passed),
                        "reason": r.reason,
                        "code_preview": code[:300],
                    }
                )
        metrics = {
            "pass1": (passed / total) if total else 0.0,
            "compile_pass_rate": (compile_ok / total) if total else 0.0,
            "num_samples": float(total),
        }
        return metrics, sample_generations

    def evaluate_suite(
        self,
        suites: Dict[str, List[Dict[str, Any]]],
        math_max_new_tokens: int,
        code_max_new_tokens: int,
    ) -> EvalResult:
        all_metrics: Dict[str, float] = {}
        merged_samples: List[Dict[str, str]] = []

        for name in ("gsm8k", "ape210k", "math", "cmath"):
            if name in suites:
                m, gens = self.evaluate_math(suites[name], max_new_tokens=math_max_new_tokens)
                for k, v in m.items():
                    all_metrics[f"{name}_{k}"] = v
                merged_samples.extend([{**x, "dataset": name} for x in gens[:2]])

        if "mbpp" in suites:
            m, gens = self.evaluate_mbpp(suites["mbpp"], max_new_tokens=code_max_new_tokens)
            for k, v in m.items():
                all_metrics[f"mbpp_{k}"] = v
            merged_samples.extend([{**x, "dataset": "mbpp"} for x in gens[:2]])

        if "gsm8k_avg_reasoning_quality" in all_metrics and "ape210k_avg_reasoning_quality" in all_metrics:
            all_metrics["avg_reasoning_quality"] = (
                all_metrics["gsm8k_avg_reasoning_quality"] + all_metrics["ape210k_avg_reasoning_quality"]
            ) / 2.0
        if "gsm8k_avg_reasoning_length_words" in all_metrics and "ape210k_avg_reasoning_length_words" in all_metrics:
            all_metrics["avg_reasoning_length_words"] = (
                all_metrics["gsm8k_avg_reasoning_length_words"] + all_metrics["ape210k_avg_reasoning_length_words"]
            ) / 2.0

        return EvalResult(metrics=all_metrics, sample_generations=merged_samples[:8])


def math_accuracy_reward(completions: List[List[Dict[str, str]]], solution: List[str], **kwargs: Any) -> List[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards: List[float] = []
    for content, sol in zip(contents, solution):
        parsed = parse_response(content)
        rewards.append(1.0 if math_answers_equal(parsed["answer"], str(sol)) else 0.0)
    return rewards


def reasoning_format_reward(completions: List[List[Dict[str, str]]], **kwargs: Any) -> List[float]:
    pattern = r"^\s*<reasoning>\s*.*?\s*</reasoning>\s*<answer>\s*.*?\s*</answer>\s*$"
    contents = [completion[0]["content"] for completion in completions]
    return [1.0 if re.match(pattern, c, flags=re.DOTALL | re.IGNORECASE) else 0.0 for c in contents]


def reasoning_length_reward(completions: List[List[Dict[str, str]]], **kwargs: Any) -> List[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards: List[float] = []
    for c in contents:
        reasoning = extract_tag_block(c, "reasoning")
        wc = approx_word_count(reasoning)
        rewards.append(min(wc / 160.0, 1.0))
    return rewards


def combined_stage1_reward(
    completions: List[List[Dict[str, str]]],
    solution: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    raw_solution: Any = solution if solution is not None else kwargs.get("solution")
    if raw_solution is None:
        raise RuntimeError(
            "GRPO reward kwargs missing `solution`. "
            "Accuracy reward cannot be computed; check dataset columns and TRL input mapping."
        )
    if not isinstance(raw_solution, (list, tuple)):
        raise TypeError(
            f"Expected `solution` to be list/tuple, got {type(raw_solution).__name__}."
        )
    solution_list = [str(x) for x in raw_solution]
    if len(solution_list) != len(completions):
        raise ValueError(
            f"`solution` length ({len(solution_list)}) does not match completions length ({len(completions)})."
        )

    acc = math_accuracy_reward(completions, solution_list, **kwargs)
    fmt = reasoning_format_reward(completions, **kwargs)
    length = reasoning_length_reward(completions, **kwargs)
    out: List[float] = []
    w_acc, w_fmt, w_len = STAGE1_REWARD_WEIGHTS
    for a, f, l in zip(acc, fmt, length):
        out.append(w_acc * a + w_fmt * f + w_len * l)
    return out


class Stage1EvalCallback(TrainerCallback):
    def __init__(
        self,
        run_dir: Path,
        quick_sets: Dict[str, List[Dict[str, Any]]],
        epoch_sets: Dict[str, List[Dict[str, Any]]],
        evaluator: OnlineEvaluator,
        eval_every_steps: int,
        math_max_new_tokens: int,
        code_max_new_tokens: int,
        early_stop_gsm_gain: float,
    ) -> None:
        self.run_dir = run_dir
        self.quick_sets = quick_sets
        self.epoch_sets = epoch_sets
        self.evaluator = evaluator
        self.eval_every_steps = max(1, eval_every_steps)
        self.math_max_new_tokens = math_max_new_tokens
        self.code_max_new_tokens = code_max_new_tokens
        self.early_stop_gsm_gain = early_stop_gsm_gain
        self.best_primary = -1.0
        self.trainer = None
        self.reward_log_history: List[float] = []
        self.quick_gsm_history: List[float] = []
        self.last_kl: Optional[float] = None
        self._is_running_eval = False

    def set_trainer(self, trainer: Any) -> None:
        self.trainer = trainer

    def _run_eval(self, eval_name: str, suites: Dict[str, List[Dict[str, Any]]], global_step: int, epoch: Optional[float]) -> Dict[str, float]:
        model = self.evaluator.model
        was_training = model.training
        model.eval()
        t0 = time.time()
        result = self.evaluator.evaluate_suite(
            suites=suites,
            math_max_new_tokens=self.math_max_new_tokens,
            code_max_new_tokens=self.code_max_new_tokens,
        )
        elapsed = time.time() - t0
        metrics = {f"{eval_name}/{k}": float(v) for k, v in result.metrics.items()}
        metrics[f"{eval_name}/elapsed_s"] = elapsed
        metrics[f"{eval_name}/global_step"] = float(global_step)
        if epoch is not None:
            metrics[f"{eval_name}/epoch"] = float(epoch)

        save_json(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "eval_name": eval_name,
                "global_step": global_step,
                "epoch": epoch,
                "metrics": result.metrics,
                "sample_generations": result.sample_generations,
            },
            self.run_dir / "eval" / f"{eval_name}_step{global_step:07d}.json",
        )
        append_jsonl(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "eval_name": eval_name,
                "global_step": global_step,
                "epoch": epoch,
                "metrics": result.metrics,
            },
            self.run_dir / "eval" / "eval_history.jsonl",
        )
        if was_training:
            model.train()
        return metrics

    def _check_reward_hacking(self) -> Optional[str]:
        if len(self.reward_log_history) < 3 or len(self.quick_gsm_history) < 3:
            return None
        r = self.reward_log_history[-3:]
        g = self.quick_gsm_history[-3:]
        reward_increasing = r[0] < r[1] < r[2]
        gsm_stagnant = abs(g[-1] - g[0]) < 0.01
        if reward_increasing and gsm_stagnant:
            return "reward_hacking_warning: reward rises for 3 evals while gsm8k accuracy is stagnant."
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        reward_val = None
        kl_val = None
        for k, v in logs.items():
            if not isinstance(v, (int, float)):
                continue
            lk = k.lower()
            if reward_val is None and "reward" in lk:
                reward_val = float(v)
            if kl_val is None and "kl" in lk:
                kl_val = float(v)
        if reward_val is not None:
            self.reward_log_history.append(reward_val)
            self.reward_log_history = self.reward_log_history[-30:]
        if kl_val is not None:
            self.last_kl = kl_val
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self._is_running_eval:
            return control
        if state.global_step <= 0 or state.global_step % self.eval_every_steps != 0:
            return control
        if self.trainer is None:
            return control

        self._is_running_eval = True
        try:
            metrics = self._run_eval("quick", self.quick_sets, state.global_step, state.epoch)
            gsm_acc = metrics.get("quick/gsm8k_accuracy")
            if gsm_acc is not None:
                self.quick_gsm_history.append(float(gsm_acc))
                self.quick_gsm_history = self.quick_gsm_history[-30:]
            self.trainer.log(metrics)

            warn = self._check_reward_hacking()
            if warn:
                append_jsonl(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "global_step": state.global_step,
                        "warning": warn,
                    },
                    self.run_dir / "eval" / "warnings.jsonl",
                )
        finally:
            self._is_running_eval = False
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self._is_running_eval:
            return control
        if self.trainer is None:
            return control
        self._is_running_eval = True
        try:
            metrics = self._run_eval("epoch", self.epoch_sets, state.global_step, state.epoch)
            self.trainer.log(metrics)

            gsm = metrics.get("epoch/gsm8k_accuracy", 0.0) * 100.0
            ape = metrics.get("epoch/ape210k_accuracy", 0.0) * 100.0
            primary = (gsm + ape) / 2.0
            gsm_gain = gsm - BASELINE_METRICS["gsm8k_accuracy"]

            if primary > self.best_primary:
                self.best_primary = primary
                best_dir = self.run_dir / "checkpoint-best"
                self.trainer.save_model(str(best_dir))
                save_json(
                    {
                        "best_primary_score": primary,
                        "global_step": state.global_step,
                        "epoch": state.epoch,
                        "metrics": metrics,
                    },
                    self.run_dir / "checkpoint-best" / "best_metrics.json",
                )

            if self.last_kl is not None and self.last_kl > 0.15:
                append_jsonl(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "global_step": state.global_step,
                        "warning": f"kl_divergence_warning: {self.last_kl:.6f} > 0.15",
                    },
                    self.run_dir / "eval" / "warnings.jsonl",
                )

            if state.epoch is not None and state.epoch >= 1.0 and gsm_gain >= self.early_stop_gsm_gain:
                control.should_training_stop = True
                save_json(
                    {
                        "reason": "early_stop_gsm_gain_reached",
                        "gsm_gain": gsm_gain,
                        "threshold": self.early_stop_gsm_gain,
                        "global_step": state.global_step,
                        "epoch": state.epoch,
                    },
                    self.run_dir / "early_stop.json",
                )
        finally:
            self._is_running_eval = False
        return control


def build_eval_suites(training_data_dir: Path, seed: int) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    files = {
        "gsm8k": training_data_dir / "gsm8k_test.json",
        "ape210k": training_data_dir / "ape210k_test.json",
        "math": training_data_dir / "math_test.json",
        "cmath": training_data_dir / "cmath_test.json",
        "mbpp": training_data_dir / "mbpp_test.json",
    }
    for name, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing eval file for {name}: {p}")

    quick = {
        "gsm8k": load_math_eval_samples(files["gsm8k"], 50, seed + 11),
        "ape210k": load_math_eval_samples(files["ape210k"], 50, seed + 12),
        "math": load_math_eval_samples(files["math"], 20, seed + 13),
        "mbpp": load_mbpp_eval_samples(files["mbpp"], 10, seed + 14),
    }
    epoch = {
        "gsm8k": load_math_eval_samples(files["gsm8k"], 500, seed + 21),
        "ape210k": load_math_eval_samples(files["ape210k"], 1000, seed + 22),
        "math": load_math_eval_samples(files["math"], 400, seed + 23),
        "cmath": load_math_eval_samples(files["cmath"], 400, seed + 24),
        "mbpp": load_mbpp_eval_samples(files["mbpp"], 100, seed + 25),
    }
    final = {
        "gsm8k": load_math_eval_samples(files["gsm8k"], None, seed + 31),
        "ape210k": load_math_eval_samples(files["ape210k"], None, seed + 32),
        "math": load_math_eval_samples(files["math"], 200, seed + 33),
        "cmath": load_math_eval_samples(files["cmath"], 100, seed + 34),
        "mbpp": load_mbpp_eval_samples(files["mbpp"], 50, seed + 35),
    }
    return quick, epoch, final


def build_success_report(final_metrics: Dict[str, float], last_kl: Optional[float]) -> Dict[str, Any]:
    gsm_acc = final_metrics.get("final/gsm8k_accuracy", 0.0) * 100.0
    ape_acc = final_metrics.get("final/ape210k_accuracy", 0.0) * 100.0
    gsm_gain = gsm_acc - BASELINE_METRICS["gsm8k_accuracy"]
    ape_gain = ape_acc - BASELINE_METRICS["ape210k_accuracy"]
    avg_rq = final_metrics.get("final/avg_reasoning_quality", 0.0)
    avg_len = final_metrics.get("final/avg_reasoning_length_words", 0.0)

    target_len = 167.0 * 1.2
    format_gsm = final_metrics.get("final/gsm8k_format_correct_rate", 0.0)
    format_ape = final_metrics.get("final/ape210k_format_correct_rate", 0.0)
    format_ok = (format_gsm >= 0.95) and (format_ape >= 0.95)

    mbpp_pass1 = final_metrics.get("final/mbpp_pass1", 0.0) * 100.0
    mbpp_drop = BASELINE_METRICS["mbpp_pass1"] - mbpp_pass1
    criteria = {
        "gsm8k_accuracy_gain_ge_7": gsm_gain >= 7.0,
        "ape210k_accuracy_gain_ge_5": ape_gain >= 5.0,
        "reasoning_quality_ge_0_75": avg_rq >= 0.75,
        "avg_reasoning_len_gain_ge_20pct": avg_len >= target_len,
        "format_correct_rate_ge_95pct": format_ok,
        "kl_divergence_le_0_15": (last_kl is not None and last_kl <= 0.15),
        "mbpp_pass1_drop_le_3": mbpp_drop <= 3.0,
    }
    ready_for_stage2 = all(criteria.values())
    return {
        "final_metrics_percent": {
            "gsm8k_accuracy": gsm_acc,
            "ape210k_accuracy": ape_acc,
            "mbpp_pass1": mbpp_pass1,
        },
        "gains": {
            "gsm8k_gain": gsm_gain,
            "ape210k_gain": ape_gain,
            "mbpp_drop": mbpp_drop,
        },
        "criteria": criteria,
        "ready_for_stage2": ready_for_stage2,
    }


def patch_vllm_api_for_trl(use_vllm: bool = False) -> bool:
    """Patch vLLM API differences so trl GRPOTrainer can run with newer vLLM versions."""
    try:
        import vllm.sampling_params as sp  # type: ignore
    except Exception as e:
        if use_vllm:
            raise ImportError(
                "vLLM is not installed (or import failed) but --use-vllm was specified. "
                "Please install vLLM or disable --use-vllm."
            ) from e
        return False

    patched = False
    needs_trl_sampling_patch = False

    if not hasattr(sp, "GuidedDecodingParams"):
        class GuidedDecodingParams:
            def __init__(self, backend: str = "outlines", regex: Optional[str] = None, **kwargs: Any):
                self.backend = backend
                self.regex = regex
                self.kwargs = kwargs

        sp.GuidedDecodingParams = GuidedDecodingParams  # type: ignore[attr-defined]
        patched = True

    sig = inspect.signature(sp.SamplingParams)
    if "guided_decoding" not in sig.parameters:
        needs_trl_sampling_patch = True

    if patched:
        print("Warning: Applied vLLM compatibility patch for TRL guided decoding API.")
    return needs_trl_sampling_patch


def patch_trl_sampling_params_for_guided_decoding() -> None:
    """Patch only TRL's local SamplingParams symbol to support guided_decoding kwarg."""
    import trl.trainer.grpo_trainer as trl_grpo_trainer  # type: ignore
    import vllm.sampling_params as sp  # type: ignore

    if getattr(trl_grpo_trainer, "_stage1_sampling_patch_applied", False):
        return

    orig_sampling_params = trl_grpo_trainer.SamplingParams
    sig = inspect.signature(orig_sampling_params)
    if "guided_decoding" in sig.parameters:
        return

    class SamplingParamsCompat:
        def __new__(cls, *args: Any, guided_decoding: Any = None, **kwargs: Any):
            # vLLM>=0.7 moved guided decoding into structured outputs.
            if guided_decoding is not None and "structured_outputs" not in kwargs:
                regex = getattr(guided_decoding, "regex", None)
                if regex and hasattr(sp, "StructuredOutputsParams"):
                    kwargs["structured_outputs"] = sp.StructuredOutputsParams(regex=regex)
            return orig_sampling_params(*args, **kwargs)

    trl_grpo_trainer.SamplingParams = SamplingParamsCompat  # type: ignore[assignment]
    trl_grpo_trainer._stage1_sampling_patch_applied = True  # type: ignore[attr-defined]
    print("Warning: Applied TRL SamplingParams compatibility patch for vLLM guided decoding.")


def import_grpo_components(use_vllm: bool = False):
    from peft import LoraConfig, TaskType

    needs_trl_sampling_patch = patch_vllm_api_for_trl(use_vllm=use_vllm)

    try:
        from trl import GRPOConfig, GRPOTrainer
        if needs_trl_sampling_patch:
            patch_trl_sampling_params_for_guided_decoding()
        return LoraConfig, TaskType, GRPOConfig, GRPOTrainer
    except (RuntimeError, ImportError, AttributeError) as e:
        err = str(e)
        # trl==0.18.0 may fail to import GRPOTrainer with newer/changed vllm APIs.
        if "GuidedDecodingParams" in err and "vllm.sampling_params" in err:
            if use_vllm:
                raise RuntimeError(
                    "Detected TRL/vLLM API incompatibility while --use-vllm is enabled. "
                    "Please align trl and vllm versions, or disable --use-vllm."
                ) from e
            import trl.import_utils as trl_import_utils

            trl_import_utils._vllm_available = False
            from trl import GRPOConfig, GRPOTrainer

            print("Warning: vLLM is incompatible with current TRL; falling back to non-vLLM GRPO path.")
            return LoraConfig, TaskType, GRPOConfig, GRPOTrainer
        raise


def ensure_vllm_colocate_env() -> None:
    """Ensure required distributed env vars exist for vLLM external launcher in single-process runs."""
    defaults = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }
    missing = [k for k in defaults if k not in os.environ]
    if not missing:
        return
    for key in missing:
        os.environ[key] = defaults[key]
    print(
        "Warning: Missing distributed env vars for vLLM colocate mode. "
        "Auto-filled for single-process run: "
        + ", ".join(f"{k}={os.environ[k]}" for k in missing)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 GRPO training for simple math.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-data", type=str, default=DEFAULT_STAGE1_TRAIN_PATH)
    parser.add_argument("--training-data-dir", type=str, default=DEFAULT_TRAINING_DATA_DIR)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="stage1_qwen2.5_3b_grpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--ape-to-gsm-ratio", type=float, default=3.0, help="Ape210K:GSM8K ratio in stage1 data.")
    parser.add_argument("--skip-rebalance", action="store_true")

    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--kl-penalty", type=float, default=0.02, help="Mapped to GRPO beta.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--math-max-new-tokens", type=int, default=512)
    parser.add_argument("--code-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=1024)

    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--early-stop-gsm-gain", type=float, default=7.0)
    parser.add_argument("--mbpp-timeout-s", type=float, default=6.0)
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-mode", type=str, default="colocate")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)

    parser.add_argument("--report-to", type=str, default="tensorboard", help="Comma-separated: tensorboard,wandb,none")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{args.run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_stage1 = load_json(Path(args.train_data))
    if not isinstance(raw_stage1, list):
        raise ValueError(f"Expected list in {args.train_data}")
    if not args.skip_rebalance:
        raw_stage1 = rebalance_stage1(raw_stage1, ape_to_gsm_ratio=args.ape_to_gsm_ratio, seed=args.seed)
    train_records = stage1_to_train_records(raw_stage1)
    if not train_records:
        raise ValueError("No valid stage1 records found after preprocessing.")

    train_dataset = Dataset.from_list(train_records)
    train_dataset = train_dataset.map(
        lambda ex: {
            "prompt": [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": ex["question"]},
            ],
            "solution": ex["solution"],
        }
    )
    if "solution" not in train_dataset.column_names:
        raise ValueError("Missing `solution` column in train dataset; reward function requires gold answers.")

    quick_sets, epoch_sets, final_sets = build_eval_suites(Path(args.training_data_dir), seed=args.seed)

    run_config = vars(args).copy()
    run_config["run_dir"] = str(run_dir)
    run_config["num_train_records"] = len(train_records)
    save_json(run_config, run_dir / "run_config.json")

    if args.dry_run:
        print("DRY RUN complete")
        print(f"Run dir: {run_dir}")
        print(f"Train records: {len(train_records)}")
        print(
            f"Quick eval sizes: gsm8k={len(quick_sets['gsm8k'])}, "
            f"ape210k={len(quick_sets['ape210k'])}, math={len(quick_sets['math'])}, mbpp={len(quick_sets['mbpp'])}"
        )
        print(
            f"Epoch eval sizes: gsm8k={len(epoch_sets['gsm8k'])}, ape210k={len(epoch_sets['ape210k'])}, "
            f"math={len(epoch_sets['math'])}, cmath={len(epoch_sets['cmath'])}, mbpp={len(epoch_sets['mbpp'])}"
        )
        return

    if args.use_vllm and not torch.cuda.is_available():
        print("Warning: CUDA is unavailable; disabling vLLM for this run.")
        args.use_vllm = False

    if args.use_vllm and args.vllm_mode == "colocate":
        ensure_vllm_colocate_env()

    try:
        LoraConfig, TaskType, GRPOConfig, GRPOTrainer = import_grpo_components(use_vllm=bool(args.use_vllm))
    except ImportError as e:
        if args.use_vllm and "vLLM" in str(e):
            raise
        raise ImportError(
            "Missing dependencies for GRPO training. Please install trl and peft in the active environment."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.config.use_cache = False

    report_to = [x.strip() for x in args.report_to.split(",") if x.strip()]
    if "none" in report_to:
        report_to = []

    cfg_kwargs: Dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "do_eval": False,
        "seed": args.seed,
        "use_vllm": bool(args.use_vllm),
        "vllm_mode": args.vllm_mode,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "num_generations": args.group_size,
        "beta": args.kl_penalty,
        "temperature": args.temperature,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.math_max_new_tokens,
        "report_to": report_to,
        "run_name": args.run_name,
        "lr_scheduler_type": "cosine",
        "remove_unused_columns": False,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
    }

    available_cfg_fields = set(GRPOConfig.__dataclass_fields__.keys())
    if args.use_vllm:
        required_vllm_field = "use_vllm"
        optional_vllm_fields = [
            "vllm_mode",
            "vllm_gpu_memory_utilization",
            "vllm_tensor_parallel_size",
        ]
        if required_vllm_field not in available_cfg_fields:
            raise ValueError(
                "Current GRPOConfig does not support `use_vllm`, so --use-vllm cannot be applied. "
                "Please upgrade/downgrade trl to a compatible version."
            )
        missing_optional = [k for k in optional_vllm_fields if k not in available_cfg_fields]
        if missing_optional:
            print(
                "Warning: Current GRPOConfig does not support these vLLM fields and they will be ignored: "
                + ", ".join(missing_optional)
            )

    filtered_cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in available_cfg_fields}
    if args.use_vllm and not bool(filtered_cfg_kwargs.get("use_vllm", False)):
        raise ValueError(
            "--use-vllm was requested, but `use_vllm` was not retained in GRPOConfig kwargs."
        )
    training_args = GRPOConfig(**filtered_cfg_kwargs)

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    reward_funcs = [combined_stage1_reward]

    evaluator = OnlineEvaluator(model=model, tokenizer=tokenizer, timeout_s=args.mbpp_timeout_s)
    eval_callback = Stage1EvalCallback(
        run_dir=run_dir,
        quick_sets=quick_sets,
        epoch_sets=epoch_sets,
        evaluator=evaluator,
        eval_every_steps=args.eval_every_steps,
        math_max_new_tokens=args.math_max_new_tokens,
        code_max_new_tokens=args.code_max_new_tokens,
        early_stop_gsm_gain=args.early_stop_gsm_gain,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    eval_callback.set_trainer(trainer)
    trainer.add_callback(eval_callback)

    checkpoint = args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(train_dataset)
    save_json(train_metrics, run_dir / "train_metrics.json")

    final_model_dir = run_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    model.eval()
    final_eval = evaluator.evaluate_suite(
        suites=final_sets,
        math_max_new_tokens=args.math_max_new_tokens,
        code_max_new_tokens=args.code_max_new_tokens,
    )
    final_metrics_prefixed = {f"final/{k}": float(v) for k, v in final_eval.metrics.items()}
    save_json(
        {
            "metrics": final_eval.metrics,
            "sample_generations": final_eval.sample_generations,
        },
        run_dir / "eval" / "final_eval.json",
    )

    decision = build_success_report(final_metrics_prefixed, eval_callback.last_kl)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "train_metrics": train_metrics,
        "final_metrics": final_eval.metrics,
        "decision": decision,
        "last_kl_seen": eval_callback.last_kl,
        "reward_weights": list(STAGE1_REWARD_WEIGHTS),
    }
    save_json(summary, run_dir / "summary.json")

    print("\nStage1 GRPO training completed")
    print(f"Run dir: {run_dir}")
    print(f"Final model: {final_model_dir}")
    print(f"Ready for stage2: {decision['ready_for_stage2']}")


if __name__ == "__main__":
    main()
