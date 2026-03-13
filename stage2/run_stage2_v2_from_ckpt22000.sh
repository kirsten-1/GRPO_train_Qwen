#!/usr/bin/env bash
set -euo pipefail

# Stage2 v2 training from checkpoint-22000 with optimized hyperparameters
# Key changes:
# - Start from checkpoint-22000 (Stage1 starting point, not overfitted)
# - lr: 1e-5 (vs 5e-6 in v1)
# - beta: 0.08 (vs 0.10 in v1)
# - format_reward weight: 1.5 (vs 0.9 in v1)
# - Train 0.5 epoch for quick validation

ROOT="${ROOT:-/root/grpo}"
export ROOT
export OPENR1_ROOT="${OPENR1_ROOT:-$ROOT/open-r1}"
export CFG="$ROOT/stage2/openr1_stage2_grpo_server_2x5090_v2.yaml"

# Use checkpoint-22000 as starting point
export RESUME_FROM_CHECKPOINT="/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090/checkpoint-22000"

# Keep vLLM launch parameters aligned with stage1-proven settings.
export TRL_VLLM_GROUP_PORT="${TRL_VLLM_GROUP_PORT:-55693}"
export TRL_VLLM_SKIP_BARRIER="${TRL_VLLM_SKIP_BARRIER:-1}"
export TRL_VLLM_SYNC_TRACE="${TRL_VLLM_SYNC_TRACE:-0}"
export TRL_VLLM_SYNC_BARRIER_INTERVAL="${TRL_VLLM_SYNC_BARRIER_INTERVAL:-0}"
export TRL_VLLM_SYNC_TRAINABLE_ONLY="${TRL_VLLM_SYNC_TRAINABLE_ONLY:-1}"
export TRL_VLLM_SYNC_PEF_TARGET_ONLY="${TRL_VLLM_SYNC_PEF_TARGET_ONLY:-1}"
export TRL_VLLM_SYNC_BEFORE_UNMERGE="${TRL_VLLM_SYNC_BEFORE_UNMERGE:-1}"
export TRL_VLLM_SKIP_UNMERGE="${TRL_VLLM_SKIP_UNMERGE:-0}"
export TRL_VLLM_SKIP_RESET_PREFIX_CACHE="${TRL_VLLM_SKIP_RESET_PREFIX_CACHE:-1}"
export TRL_VLLM_UPDATE_STRICT="${TRL_VLLM_UPDATE_STRICT:-0}"
export TRL_VLLM_REQUEST_TIMEOUT="${TRL_VLLM_REQUEST_TIMEOUT:-240}"
export TRL_VLLM_GENERATE_RECV_TIMEOUT="${TRL_VLLM_GENERATE_RECV_TIMEOUT:-120}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-warning}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"
export SERVER_GPU="${SERVER_GPU:-1}"
export TRAIN_GPU="${TRAIN_GPU:-0}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
# Workaround for vLLM+PyTorch duplicate template assertion on some nodes.
export VLLM_USE_STANDALONE_COMPILE="${VLLM_USE_STANDALONE_COMPILE:-0}"

echo "=== Stage2 v2 Training Configuration ==="
echo "Config: $CFG"
echo "Resume from: $RESUME_FROM_CHECKPOINT"
echo "Training: 0.5 epoch (~1,744 steps)"
echo "Expected duration: ~2 hours"
echo "Key changes: lr=1e-5, beta=0.08, format_weight=1.5"
echo "========================================"
echo ""

if printf '%s\n' "$@" | grep -qF -- '--resume_from_checkpoint'; then
  exec bash "$ROOT/stage1/run_stage1_openr1_server.sh" "$@"
else
  exec bash "$ROOT/stage1/run_stage1_openr1_server.sh" \
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
    "$@"
fi
