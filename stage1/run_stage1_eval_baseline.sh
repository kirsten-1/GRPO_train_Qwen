#!/usr/bin/env bash
set -euo pipefail

# Baseline eval should default to the untouched base model.
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/stage1_eval_runs}"
CODE_BENCHMARKS="${CODE_BENCHMARKS:-none}"  # none|mbpp|mbpp_humaneval|all

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python /root/grpo/stage1/eval_stage1_model.py \
  --model-path "$MODEL_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "stage1_baseline_eval" \
  --suite baseline \
  --code-benchmarks "$CODE_BENCHMARKS"
