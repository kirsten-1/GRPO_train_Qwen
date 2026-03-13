#!/usr/bin/env bash
set -euo pipefail

# Stage2 v2 training from checkpoint-22500 with optimized hyperparameters
# 
# Strategy:
# - Start from checkpoint-22500 (verified best: MATH 38.0%, GSM8K 54.5%)
# - Train 0.5 epoch (~1,744 steps, ~2 hours)
# - Save every 250 steps (7 evaluation points)
# - Monitor format_reward (target: 44% -> 60%+)
# - Monitor accuracy_reward (target: maintain 55%+)
#
# Key changes from Stage2 v1:
# - lr: 5e-6 -> 1e-5 (Stage1 proven rate)
# - beta: 0.10 -> 0.08 (more exploration space)
# - format_weight: 0.9 -> 1.5 (prioritize format learning)
# - accuracy_weight: 1.3 -> 1.2 (reduce hard task penalty)
#
# Early stopping conditions:
# - ✅ Continue: format_reward > 50% at step 250
# - ⚠️  Observe: format_reward 45-50% at step 250, check step 500
# - ❌ Stop: format_reward < 45% at step 250 OR GSM8K drops > 2%

ROOT="${ROOT:-/root/grpo}"
export ROOT
export OPENR1_ROOT="${OPENR1_ROOT:-$ROOT/open-r1}"
export CFG="$ROOT/stage2/openr1_stage2_grpo_server_2x5090_v2.yaml"

# Use checkpoint-22500 as starting point (verified best)
export RESUME_FROM_CHECKPOINT="/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090/checkpoint-22500"

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

echo "=========================================="
echo "  Stage2 v2 Training - Optimized Config"
echo "=========================================="
echo ""
echo "Config: $CFG"
echo "Resume from: $RESUME_FROM_CHECKPOINT"
echo ""
echo "Training Plan:"
echo "  - Duration: 0.5 epoch (~1,744 steps, ~2 hours)"
echo "  - Checkpoints: Every 250 steps (7 total)"
echo "  - Baseline: MATH 38.0%, GSM8K 54.5%"
echo ""
echo "Key Optimizations:"
echo "  - Learning rate: 1e-5 (was 5e-6)"
echo "  - KL penalty (beta): 0.08 (was 0.10)"
echo "  - Format reward weight: 1.5 (was 0.9) ← KEY FIX"
echo "  - Accuracy reward weight: 1.2 (was 1.3)"
echo ""
echo "Success Criteria (at 0.5 epoch):"
echo "  - Format reward ≥ 60% (baseline: 44.05%)"
echo "  - Accuracy reward ≥ 60% (baseline: 55.07%)"
echo "  - GSM8K ≥ 53.5% (baseline: 54.5%, allow -1%)"
echo "  - MATH ≥ 40% (baseline: 38.0%, target +2%)"
echo ""
echo "Early Stop Triggers:"
echo "  - Format reward < 45% at step 250"
echo "  - GSM8K drops > 2% at any checkpoint"
echo "  - Accuracy reward < 45% at step 500"
echo ""
echo "=========================================="
echo ""
read -p "Press Enter to start training, or Ctrl+C to cancel..."

if printf '%s\n' "$@" | grep -qF -- '--resume_from_checkpoint'; then
  exec bash "$ROOT/stage1/run_stage1_openr1_server.sh" "$@"
else
  exec bash "$ROOT/stage1/run_stage1_openr1_server.sh" \
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
    "$@"
fi
