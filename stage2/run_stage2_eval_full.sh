#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/stage2_eval_runs}"
SUITE="${SUITE:-full}"
CODE_NUM_SAMPLES="${CODE_NUM_SAMPLES:-5}"
TIMEOUT_S="${TIMEOUT_S:-6}"
CODE_TEMPERATURE="${CODE_TEMPERATURE:-0.7}"
CODE_TOP_P="${CODE_TOP_P:-0.95}"
MBPP_SAMPLES="${MBPP_SAMPLES:-}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

ARGS=(
  --model-path "$MODEL_PATH"
  --base-model-path "$BASE_MODEL_PATH"
  --output-root "$OUTPUT_ROOT"
  --suite "$SUITE"
  --code-num-samples "$CODE_NUM_SAMPLES"
  --timeout-s "$TIMEOUT_S"
  --temperature "$CODE_TEMPERATURE"
  --top-p "$CODE_TOP_P"
  --progress-every "$PROGRESS_EVERY"
  --run-name stage2_eval_full
)

if [[ -n "$MBPP_SAMPLES" ]]; then
  ARGS+=(--mbpp-samples "$MBPP_SAMPLES")
fi
if [[ -n "$HUMANEVAL_SAMPLES" ]]; then
  ARGS+=(--humaneval-samples "$HUMANEVAL_SAMPLES")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python /root/grpo/stage2/eval_stage2_model.py "${ARGS[@]}"
