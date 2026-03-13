#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/stage1_eval_runs}"
CODE_BENCHMARKS="${CODE_BENCHMARKS:-none}"  # none|mbpp|mbpp_humaneval|all

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python /root/grpo/stage1/eval_stage1_model.py \
  --model-path "$MODEL_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "stage1_smoke_eval" \
  --suite smoke \
  --progress-every 1 \
  --code-benchmarks "$CODE_BENCHMARKS"
