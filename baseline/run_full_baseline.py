#!/usr/bin/env python3
"""
全量 baseline 评估脚本（Qwen2.5-3B Instruct）

覆盖任务：
- 数学：GSM8K / Ape210K / MATH / CMATH
- 代码：MBPP / HumanEval / APPS

输出：
- 每个数据集的指标 JSON
- 汇总指标 summary JSON
- 每个数据集的典型失败案例 JSON

说明：
- 模型路径已硬编码为本地路径，禁止在线下载。
- 默认离线模式（datasets/transformers）。
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
import importlib.util
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
from typing import Any, Dict, List, Optional, Tuple

# 强制离线，避免运行时走网络；需在导入 HF 相关库前设置
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from datasets import DownloadConfig, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 硬编码模型路径，防止基线运行时再次下载模型
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct"
HF_LOCAL_ONLY_DOWNLOAD_CONFIG = DownloadConfig(local_files_only=True)

PROCESSED_TEST_FILES = {
    "gsm8k": "/root/autodl-tmp/training_data/gsm8k_test.json",
    "ape210k": "/root/autodl-tmp/training_data/ape210k_test.json",
    "math": "/root/autodl-tmp/training_data/math_test.json",
    "cmath": "/root/autodl-tmp/training_data/cmath_test.json",
    "mbpp": "/root/autodl-tmp/training_data/mbpp_test.json",
    "humaneval": "/root/autodl-tmp/training_data/humaneval_test.json",
}

APPS_TEST_ROOT = "/root/autodl-tmp/datasets/APPS/test"
DATASETS_CACHE_DIR = "/root/autodl-tmp/datasets"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/baseline_results"

SYSTEM_PROMPT_RECOMMENDED = """你是一个专业的AI助手，擅长数学推理和编程。请一步步思考并解决问题。

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


@dataclass
class GenerationStats:
    gen_tokens: int
    elapsed_s: float


@dataclass
class CodeEvalResult:
    compile_ok: bool
    passed: bool
    reason: str
    runtime_s: float
    details: str = ""


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
    if m:
        return m.group(1).strip()
    return code


def approx_word_count(text: str) -> int:
    # 中文按单字计数，英文按 token 粗略计数
    tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text)
    return len(tokens)


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

    # mixed fraction: 4(5/11)
    mm = re.fullmatch(r"(-?\d+)\((\d+)/(\d+)\)", expr)
    if mm:
        whole = float(mm.group(1))
        num = float(mm.group(2))
        den = float(mm.group(3))
        if den == 0:
            return None
        sign = -1.0 if whole < 0 else 1.0
        return whole + sign * (num / den)

    # percentage: 150%
    if expr.endswith("%"):
        base = safe_eval_math_expr(expr[:-1])
        return None if base is None else base / 100.0

    # 允许简单算术表达式
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


def normalize_io_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()


def output_matches(actual: str, expected: Any) -> bool:
    actual_norm = normalize_io_text(actual)
    if isinstance(expected, list):
        return any(actual_norm == normalize_io_text(str(x)) for x in expected)
    return actual_norm == normalize_io_text(str(expected))


def function_output_matches(actual: str, expected: Any) -> bool:
    actual_s = normalize_io_text(actual)
    try:
        actual_obj = json.loads(actual_s)
    except Exception:
        return output_matches(actual_s, expected)

    expected_obj = expected
    if isinstance(expected, str):
        try:
            expected_obj = json.loads(expected)
        except Exception:
            try:
                expected_obj = ast.literal_eval(expected)
            except Exception:
                expected_obj = expected

    return actual_obj == expected_obj


def run_python_code(code: str, stdin_text: str = "", timeout_s: float = 5.0) -> Tuple[bool, subprocess.CompletedProcess[str], float, str]:
    code = strip_code_fences(code)
    with tempfile.TemporaryDirectory(prefix="baseline_eval_") as td:
        td_path = Path(td)
        script_path = td_path / "main.py"
        script_path.write_text(code, encoding="utf-8")

        # compile
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
    return CodeEvalResult(compile_ok=compile_ok, passed=passed, reason=("pass" if passed else reason), runtime_s=elapsed, details=details)


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
    return CodeEvalResult(compile_ok=compile_ok, passed=passed, reason=("pass" if passed else reason), runtime_s=elapsed, details=details)


def eval_apps_code_stdio(code: str, io_pairs: List[Tuple[Any, Any]], timeout_s: float, max_cases: Optional[int]) -> CodeEvalResult:
    compile_ok, _, _, compile_reason = run_python_code(strip_code_fences(code), timeout_s=timeout_s)
    if not compile_ok:
        return CodeEvalResult(compile_ok=False, passed=False, reason=compile_reason, runtime_s=0.0)

    total_runtime = 0.0
    cases = io_pairs if max_cases is None else io_pairs[:max_cases]
    for inp, expected in cases:
        stdin_text = str(inp)
        c_ok, proc, elapsed, reason = run_python_code(strip_code_fences(code), stdin_text=stdin_text, timeout_s=timeout_s)
        total_runtime += elapsed
        if not c_ok:
            return CodeEvalResult(compile_ok=False, passed=False, reason="compile_error", runtime_s=total_runtime)
        if reason in {"runtime_error", "timeout"}:
            return CodeEvalResult(compile_ok=True, passed=False, reason=reason, runtime_s=total_runtime, details=(proc.stderr[-300:] if proc.stderr else ""))
        if not output_matches(proc.stdout, expected):
            detail = f"expected={str(expected)[:180]} | got={proc.stdout[:180]}"
            return CodeEvalResult(compile_ok=True, passed=False, reason="wrong_output", runtime_s=total_runtime, details=detail)

    return CodeEvalResult(compile_ok=True, passed=True, reason="pass", runtime_s=total_runtime)


def eval_apps_code_functional(code: str, fn_name: str, io_pairs: List[Tuple[Any, Any]], timeout_s: float, max_cases: Optional[int]) -> CodeEvalResult:
    compile_ok, _, _, compile_reason = run_python_code(strip_code_fences(code), timeout_s=timeout_s)
    if not compile_ok:
        return CodeEvalResult(compile_ok=False, passed=False, reason=compile_reason, runtime_s=0.0)

    total_runtime = 0.0
    cases = io_pairs if max_cases is None else io_pairs[:max_cases]
    for inp, expected in cases:
        harness = (
            f"{strip_code_fences(code)}\n\n"
            "import ast, json\n"
            f"_fn = {fn_name}\n"
            f"_raw_inp = {repr(inp)}\n"
            "if isinstance(_raw_inp, str):\n"
            "    try:\n"
            "        _parsed = ast.literal_eval(_raw_inp)\n"
            "    except Exception:\n"
            "        _parsed = _raw_inp\n"
            "else:\n"
            "    _parsed = _raw_inp\n"
            "if isinstance(_parsed, tuple):\n"
            "    _res = _fn(*_parsed)\n"
            "elif isinstance(_parsed, list):\n"
            "    _res = _fn(*_parsed)\n"
            "else:\n"
            "    _res = _fn(_parsed)\n"
            "print(json.dumps(_res, ensure_ascii=False))\n"
        )
        c_ok, proc, elapsed, reason = run_python_code(harness, timeout_s=timeout_s)
        total_runtime += elapsed
        if not c_ok:
            return CodeEvalResult(compile_ok=False, passed=False, reason="compile_error", runtime_s=total_runtime)
        if reason in {"runtime_error", "timeout"}:
            return CodeEvalResult(compile_ok=True, passed=False, reason=reason, runtime_s=total_runtime, details=(proc.stderr[-300:] if proc.stderr else ""))
        if not function_output_matches(proc.stdout, expected):
            detail = f"expected={str(expected)[:180]} | got={proc.stdout[:180]}"
            return CodeEvalResult(compile_ok=True, passed=False, reason="wrong_output", runtime_s=total_runtime, details=detail)

    return CodeEvalResult(compile_ok=True, passed=True, reason="pass", runtime_s=total_runtime)


def parse_mbpp_tests_from_user(user_text: str) -> List[str]:
    marker = "测试用例："
    if marker not in user_text:
        return []
    block = user_text.split(marker, 1)[1]
    return [line.strip() for line in block.splitlines() if line.strip()]


def parse_humaneval_from_user(user_text: str) -> Tuple[str, str]:
    marker = "测试用例："
    if marker in user_text:
        prompt, test = user_text.split(marker, 1)
        return prompt.strip(), test.strip()
    return user_text.strip(), ""


class LocalChatGenerator:
    def __init__(self) -> None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"本地模型不存在: {MODEL_PATH}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        has_accelerate = importlib.util.find_spec("accelerate") is not None
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
            "dtype": dtype,
        }
        if torch.cuda.is_available() and has_accelerate:
            model_kwargs["device_map"] = "auto"
        elif torch.cuda.is_available() and not has_accelerate:
            print("Warning: accelerate not found; falling back to single-device CUDA loading.")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
        except TypeError:
            # Backward compatibility for older transformers versions that still use `torch_dtype`.
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = dtype
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)

        if torch.cuda.is_available() and "device_map" not in model_kwargs:
            self.model = self.model.to("cuda")
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Tuple[List[str], GenerationStats]:
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
            if prompt_ids is None:
                raise ValueError("chat template 输出中缺少 input_ids")
            attention_mask = prompt_inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(prompt_ids)
        else:
            raise TypeError(f"未知的 chat template 输出类型: {type(prompt_inputs)}")

        prompt_ids = prompt_ids.to(self.input_device)
        attention_mask = attention_mask.to(self.input_device)

        if not do_sample:
            temperature = 1.0

        t0 = time.perf_counter()
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
        elapsed = time.perf_counter() - t0

        prompt_len = prompt_ids.shape[-1]
        texts: List[str] = []
        gen_tokens = 0
        for seq in outputs:
            generated = seq[prompt_len:]
            gen_tokens += int(generated.shape[-1])
            texts.append(self.tokenizer.decode(generated, skip_special_tokens=True))

        return texts, GenerationStats(gen_tokens=gen_tokens, elapsed_s=elapsed)


def load_processed_samples(dataset_name: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    path = PROCESSED_TEST_FILES[dataset_name]
    data = load_json(path)
    samples: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        messages = item["messages"]
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        gold_assistant = messages[2]["content"]
        gold_answer = extract_tag_block(gold_assistant, "answer")
        gold_reasoning = extract_tag_block(gold_assistant, "reasoning")
        sample = {
            "id": item.get("id", f"{dataset_name}_{i}"),
            "dataset": dataset_name,
            "task_type": item.get("task_type", "math"),
            "system": system_prompt,
            "user": user_prompt,
            "gold_answer": gold_answer,
            "gold_reasoning": gold_reasoning,
            "difficulty": item.get("difficulty", "unknown"),
        }
        samples.append(sample)

    # MATH / CMATH 尝试补充难度信息（按索引对齐原始 test split）
    if dataset_name == "math":
        try:
            raw = load_dataset(
                "EleutherAI/hendrycks_math",
                "algebra",
                cache_dir=DATASETS_CACHE_DIR,
                download_config=HF_LOCAL_ONLY_DOWNLOAD_CONFIG,
            )["test"]
            if len(raw) == len(samples):
                for i in range(len(samples)):
                    samples[i]["difficulty"] = raw[i].get("level", "unknown")
        except Exception as e:
            print(f"Warning: 无法从本地缓存加载 hendrycks_math 难度信息，跳过。{e}")
    if dataset_name == "cmath":
        try:
            raw = load_dataset(
                "weitianwen/cmath",
                cache_dir=DATASETS_CACHE_DIR,
                download_config=HF_LOCAL_ONLY_DOWNLOAD_CONFIG,
            )["test"]
            if len(raw) == len(samples):
                for i in range(len(samples)):
                    samples[i]["difficulty"] = f"grade_{raw[i].get('grade', 'unknown')}"
        except Exception as e:
            print(f"Warning: 无法从本地缓存加载 cmath 难度信息，跳过。{e}")

    if max_samples is not None and max_samples > 0 and len(samples) > max_samples:
        samples = samples[:max_samples]
    return samples


def load_apps_samples(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    root = Path(APPS_TEST_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"APPS 测试目录不存在: {root}")

    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    samples: List[Dict[str, Any]] = []
    for d in dirs:
        if max_samples is not None and max_samples > 0 and len(samples) >= max_samples:
            break
        qf = d / "question.txt"
        iof = d / "input_output.json"
        if not qf.exists() or not iof.exists():
            continue
        try:
            question = qf.read_text(encoding="utf-8").strip()
            io_data = json.loads(iof.read_text(encoding="utf-8"))
            inputs = io_data.get("inputs", [])
            outputs = io_data.get("outputs", [])
            if not isinstance(inputs, list) or not isinstance(outputs, list):
                continue
            if len(inputs) == 0 or len(inputs) != len(outputs):
                continue

            meta_path = d / "metadata.json"
            difficulty = "unknown"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    difficulty = str(meta.get("difficulty", "unknown"))
                except Exception:
                    difficulty = "unknown"

            sample = {
                "id": f"apps_test_{d.name}",
                "dataset": "apps",
                "task_type": "code",
                "system": SYSTEM_PROMPT_RECOMMENDED,
                "user": question,
                "gold_answer": "",
                "gold_reasoning": "",
                "difficulty": difficulty,
                "fn_name": io_data.get("fn_name"),
                "io_path": str(iof),
                "num_test_cases": len(inputs),
            }
            samples.append(sample)
        except Exception:
            continue

    return samples


def load_apps_io_pairs(io_path: str, max_cases: Optional[int]) -> List[Tuple[Any, Any]]:
    io_data = json.loads(Path(io_path).read_text(encoding="utf-8"))
    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return []
    pairs = list(zip(inputs, outputs))
    return pairs if max_cases is None else pairs[:max_cases]


def evaluate_math_dataset(
    dataset_name: str,
    samples: List[Dict[str, Any]],
    generator: LocalChatGenerator,
    max_new_tokens: int,
    max_failure_cases: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    total = len(samples)
    correct = 0
    format_ok_cnt = 0
    reasoning_quality_sum = 0.0
    reasoning_wc_sum = 0
    reasoning_char_sum = 0
    gen_token_sum = 0
    gen_time_sum = 0.0
    failures: List[Dict[str, Any]] = []

    by_diff: Dict[str, Dict[str, int]] = {}

    pbar = tqdm(samples, desc=f"{dataset_name} (math baseline)")
    for sample in pbar:
        outs, gen_stats = generator.generate(
            system_prompt=sample["system"],
            user_prompt=sample["user"],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
        )
        text = outs[0]
        parsed = parse_response(text)
        pred_answer = parsed["answer"]
        gold_answer = sample["gold_answer"]
        is_correct = math_answers_equal(pred_answer, gold_answer)

        if is_correct:
            correct += 1
        if parsed["format_ok"]:
            format_ok_cnt += 1

        rq = reasoning_quality_score(parsed["reasoning"])
        rw = approx_word_count(parsed["reasoning"])
        rc = len(parsed["reasoning"])
        reasoning_quality_sum += rq
        reasoning_wc_sum += rw
        reasoning_char_sum += rc

        gen_token_sum += gen_stats.gen_tokens
        gen_time_sum += gen_stats.elapsed_s

        diff = str(sample.get("difficulty", "unknown"))
        if diff not in by_diff:
            by_diff[diff] = {"total": 0, "correct": 0}
        by_diff[diff]["total"] += 1
        if is_correct:
            by_diff[diff]["correct"] += 1

        if (not is_correct or not parsed["format_ok"]) and len(failures) < max_failure_cases:
            failures.append(
                {
                    "id": sample["id"],
                    "difficulty": diff,
                    "question": sample["user"][:600],
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "format_ok": parsed["format_ok"],
                    "reasoning_preview": parsed["reasoning"][:500],
                    "response_preview": text[:1000],
                    "fail_type": "wrong_answer" if not is_correct else "format_error",
                }
            )

    acc = correct / total if total else 0.0
    metrics = {
        "dataset": dataset_name,
        "task_type": "math",
        "num_samples": total,
        "accuracy": acc,
        "pass@1": acc,
        "format_correct_rate": format_ok_cnt / total if total else 0.0,
        "avg_reasoning_quality": reasoning_quality_sum / total if total else 0.0,
        "avg_reasoning_length_words": reasoning_wc_sum / total if total else 0.0,
        "avg_reasoning_length_chars": reasoning_char_sum / total if total else 0.0,
        "tokens_per_second": gen_token_sum / gen_time_sum if gen_time_sum > 0 else 0.0,
        "total_generated_tokens": gen_token_sum,
        "total_generation_time_s": gen_time_sum,
        "accuracy_by_difficulty": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": (v["correct"] / v["total"]) if v["total"] else 0.0,
            }
            for k, v in sorted(by_diff.items(), key=lambda x: x[0])
        },
        "notes": "推理质量为启发式分数（长度/步骤结构/验证语句）。",
    }
    return metrics, failures


def evaluate_code_dataset(
    dataset_name: str,
    samples: List[Dict[str, Any]],
    generator: LocalChatGenerator,
    code_num_samples: int,
    code_max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
    apps_max_cases: Optional[int],
    max_failure_cases: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    total = len(samples)
    pass1 = pass5 = pass10 = 0
    format_ok_cnt = 0
    compile_ok_total = 0
    passed_total = 0
    candidate_total = 0
    code_line_sum = 0
    runtime_sum = 0.0
    runtime_cnt = 0
    gen_token_sum = 0
    gen_time_sum = 0.0
    failures: List[Dict[str, Any]] = []
    by_diff: Dict[str, Dict[str, int]] = {}

    pbar = tqdm(samples, desc=f"{dataset_name} (code baseline)")
    for sample in pbar:
        apps_io_pairs: List[Tuple[Any, Any]] = []
        apps_fn_name: Optional[str] = None
        if dataset_name == "apps":
            apps_io_pairs = sample.get("io_pairs", [])
            if not apps_io_pairs:
                io_path = sample.get("io_path")
                if io_path:
                    apps_io_pairs = load_apps_io_pairs(io_path, max_cases=apps_max_cases)
            apps_fn_name = str(sample.get("fn_name")) if sample.get("fn_name") else None

        outs, gen_stats = generator.generate(
            system_prompt=sample["system"],
            user_prompt=sample["user"],
            max_new_tokens=code_max_new_tokens,
            num_return_sequences=code_num_samples,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        gen_token_sum += gen_stats.gen_tokens
        gen_time_sum += gen_stats.elapsed_s

        results: List[CodeEvalResult] = []
        for out in outs:
            parsed = parse_response(out)
            if parsed["format_ok"]:
                format_ok_cnt += 1

            code = strip_code_fences(parsed["answer"])
            code_line_sum += len([ln for ln in code.splitlines() if ln.strip()])

            if dataset_name == "mbpp":
                tests = parse_mbpp_tests_from_user(sample["user"])
                r = eval_mbpp_code(code, tests, timeout_s=timeout_s)
            elif dataset_name == "humaneval":
                prompt, test_code = parse_humaneval_from_user(sample["user"])
                r = eval_humaneval_code(code, prompt=prompt, test_code=test_code, timeout_s=timeout_s)
            elif dataset_name == "apps":
                if not apps_io_pairs:
                    r = CodeEvalResult(compile_ok=False, passed=False, reason="missing_io_pairs", runtime_s=0.0)
                else:
                    if apps_fn_name:
                        r = eval_apps_code_functional(
                            code,
                            fn_name=apps_fn_name,
                            io_pairs=apps_io_pairs,
                            timeout_s=timeout_s,
                            max_cases=apps_max_cases,
                        )
                    else:
                        r = eval_apps_code_stdio(code, io_pairs=apps_io_pairs, timeout_s=timeout_s, max_cases=apps_max_cases)
            else:
                r = CodeEvalResult(compile_ok=False, passed=False, reason="unsupported_dataset", runtime_s=0.0)

            results.append(r)
            compile_ok_total += 1 if r.compile_ok else 0
            passed_total += 1 if r.passed else 0
            candidate_total += 1
            if r.runtime_s > 0:
                runtime_sum += r.runtime_s
                runtime_cnt += 1

        top1 = results[:1]
        top5 = results[: min(5, len(results))]
        top10 = results[: min(10, len(results))]
        ok1 = any(r.passed for r in top1)
        ok5 = any(r.passed for r in top5)
        ok10 = any(r.passed for r in top10)
        pass1 += 1 if ok1 else 0
        pass5 += 1 if ok5 else 0
        pass10 += 1 if ok10 else 0

        diff = str(sample.get("difficulty", "unknown"))
        if diff not in by_diff:
            by_diff[diff] = {"total": 0, "pass1": 0, "pass10": 0}
        by_diff[diff]["total"] += 1
        if ok1:
            by_diff[diff]["pass1"] += 1
        if ok10:
            by_diff[diff]["pass10"] += 1

        if not ok10 and len(failures) < max_failure_cases:
            first = results[0] if results else CodeEvalResult(False, False, "no_output", 0.0)
            failures.append(
                {
                    "id": sample["id"],
                    "difficulty": sample.get("difficulty", "unknown"),
                    "question": sample["user"][:800],
                    "first_fail_reason": first.reason,
                    "first_fail_details": first.details[:600],
                    "candidate_reasons": [r.reason for r in results],
                }
            )

        pbar.set_postfix(pass1=f"{pass1/max(1,total):.3f}", pass10=f"{pass10/max(1,total):.3f}")

    metrics = {
        "dataset": dataset_name,
        "task_type": "code",
        "num_samples": total,
        "num_generations_per_sample": code_num_samples,
        "pass@1": pass1 / total if total else 0.0,
        "pass@5": pass5 / total if total else 0.0,
        "pass@10": pass10 / total if total else 0.0,
        "compile_pass_rate": compile_ok_total / candidate_total if candidate_total else 0.0,
        "test_pass_rate": passed_total / candidate_total if candidate_total else 0.0,
        "format_correct_rate": format_ok_cnt / candidate_total if candidate_total else 0.0,
        "avg_code_lines": code_line_sum / candidate_total if candidate_total else 0.0,
        "avg_runtime_s": runtime_sum / runtime_cnt if runtime_cnt else 0.0,
        "tokens_per_second": gen_token_sum / gen_time_sum if gen_time_sum > 0 else 0.0,
        "total_generated_tokens": gen_token_sum,
        "total_generation_time_s": gen_time_sum,
    }
    if dataset_name == "apps":
        metrics["runtime_efficiency_note"] = "avg_runtime_s 为候选代码在测试执行中的平均耗时。"
        metrics["pass_by_difficulty"] = {
            k: {
                "total": v["total"],
                "pass@1": (v["pass1"] / v["total"]) if v["total"] else 0.0,
                "pass@10": (v["pass10"] / v["total"]) if v["total"] else 0.0,
            }
            for k, v in sorted(by_diff.items(), key=lambda x: x[0])
        }
    return metrics, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="全量 baseline 评估（本地 Qwen2.5-3B）")
    parser.add_argument(
        "--datasets",
        type=str,
        default="gsm8k,ape210k,math,cmath,mbpp,humaneval,apps",
        help="逗号分隔的数据集名",
    )
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)

    # 采样控制
    parser.add_argument("--ape-max-samples", type=int, default=5000, help="Ape210K测试采样数，<=0 表示全量")
    parser.add_argument("--apps-max-samples", type=int, default=0, help="APPS 测试采样数，<=0 表示全量")
    parser.add_argument("--apps-max-test-cases", type=int, default=0, help="每题最多测试用例数，<=0 表示全量")

    # 生成参数
    parser.add_argument("--math-max-new-tokens", type=int, default=512)
    parser.add_argument("--code-max-new-tokens", type=int, default=1024)
    parser.add_argument("--code-num-samples", type=int, default=20, help="用于 Pass@k 的采样数量（建议 >=20）")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--code-timeout-s", type=float, default=6.0, help="单次代码执行超时秒数")
    parser.add_argument("--max-failure-cases", type=int, default=120)

    parser.add_argument("--dry-run", action="store_true", help="仅加载数据并打印规模，不加载模型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    selected = [x.strip() for x in args.datasets.split(",") if x.strip()]
    allowed = {"gsm8k", "ape210k", "math", "cmath", "mbpp", "humaneval", "apps"}
    invalid = [x for x in selected if x not in allowed]
    if invalid:
        raise ValueError(f"未知数据集: {invalid}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"baseline_qwen2.5-3b_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    apps_max_cases = args.apps_max_test_cases if args.apps_max_test_cases > 0 else None
    ape_max_samples = args.ape_max_samples if args.ape_max_samples > 0 else None
    apps_max_samples = args.apps_max_samples if args.apps_max_samples > 0 else None

    # 数据加载
    datasets_data: Dict[str, List[Dict[str, Any]]] = {}
    for name in selected:
        if name == "apps":
            datasets_data[name] = load_apps_samples(max_samples=apps_max_samples)
        elif name == "ape210k":
            datasets_data[name] = load_processed_samples(name, max_samples=ape_max_samples)
        else:
            datasets_data[name] = load_processed_samples(name)

    if args.dry_run:
        print("DRY RUN 数据规模：")
        for name in selected:
            data = datasets_data[name]
            print(f"  - {name}: {len(data)}")
        print(f"模型路径(硬编码): {MODEL_PATH}")
        print(f"输出目录: {run_dir}")
        return

    # 保存配置
    run_config = {
        "model_path": MODEL_PATH,
        "selected_datasets": selected,
        "ape_max_samples": ape_max_samples,
        "apps_max_samples": apps_max_samples,
        "apps_max_test_cases": apps_max_cases,
        "math_max_new_tokens": args.math_max_new_tokens,
        "code_max_new_tokens": args.code_max_new_tokens,
        "code_num_samples": args.code_num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "code_timeout_s": args.code_timeout_s,
        "max_failure_cases": args.max_failure_cases,
        "seed": args.seed,
        "timestamp_utc": ts,
    }
    save_json(run_config, run_dir / "run_config.json")

    generator = LocalChatGenerator()

    summary: Dict[str, Any] = {"model_path": MODEL_PATH, "datasets": {}, "timestamp_utc": ts}
    all_failures: Dict[str, List[Dict[str, Any]]] = {}

    for name in selected:
        samples = datasets_data[name]
        if not samples:
            print(f"⚠️ 跳过 {name}: 数据为空")
            continue

        if name in {"gsm8k", "ape210k", "math", "cmath"}:
            metrics, failures = evaluate_math_dataset(
                dataset_name=name,
                samples=samples,
                generator=generator,
                max_new_tokens=args.math_max_new_tokens,
                max_failure_cases=args.max_failure_cases,
            )
        else:
            metrics, failures = evaluate_code_dataset(
                dataset_name=name,
                samples=samples,
                generator=generator,
                code_num_samples=args.code_num_samples,
                code_max_new_tokens=args.code_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout_s=args.code_timeout_s,
                apps_max_cases=apps_max_cases,
                max_failure_cases=args.max_failure_cases,
            )

        summary["datasets"][name] = metrics
        all_failures[name] = failures
        save_json(metrics, run_dir / f"metrics_{name}.json")
        save_json(failures, run_dir / f"failures_{name}.json")
        print(f"✅ {name} 评估完成 -> {run_dir / f'metrics_{name}.json'}")

    # 汇总 tokens/s（简单宏平均 + 加权平均）
    tps_values = []
    weighted_tokens = 0
    weighted_time = 0.0
    for _, m in summary["datasets"].items():
        tps = m.get("tokens_per_second", 0.0)
        toks = m.get("total_generated_tokens", 0)
        ts_s = m.get("total_generation_time_s", 0.0)
        if tps > 0:
            tps_values.append(tps)
        weighted_tokens += toks
        weighted_time += ts_s

    summary["global"] = {
        "datasets_evaluated": list(summary["datasets"].keys()),
        "avg_tokens_per_second_macro": (sum(tps_values) / len(tps_values)) if tps_values else 0.0,
        "tokens_per_second_weighted": (weighted_tokens / weighted_time) if weighted_time > 0 else 0.0,
        "total_generated_tokens": weighted_tokens,
        "total_generation_time_s": weighted_time,
    }

    save_json(summary, run_dir / "summary.json")
    save_json(all_failures, run_dir / "failure_cases_all.json")

    print("\n================ BASELINE 完成 ================")
    print(f"模型: {MODEL_PATH}")
    print(f"结果目录: {run_dir}")
    print("输出文件:")
    print("  - run_config.json")
    print("  - summary.json")
    print("  - metrics_<dataset>.json")
    print("  - failures_<dataset>.json")
    print("  - failure_cases_all.json")


if __name__ == "__main__":
    main()
