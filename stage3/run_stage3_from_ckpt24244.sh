#!/usr/bin/env bash
set -euo pipefail

# Stage3 training from Stage2 v2 checkpoint-24244
#
# Data: 51,888 samples (36,888 code + 15,000 math)
# Duration: 0.5 epoch (~6,486 steps, ~10 hours)
# Checkpoints: Every 500 steps (13 total)
#
# Baseline (ckpt-24244):
#   MATH 35.9%, GSM8K 55.5%, MBPP 62.5%, HumanEval 53.1%, Format 91.0%
#
# Goals:
#   - MBPP pass@1 >= 65% (baseline 62.5%)
#   - HumanEval pass@1 >= 60% (baseline 53.1%)
#   - GSM8K >= 50% (baseline 55.5%, allow -5%)
#   - MATH >= 33% (baseline 35.9%, allow -3%)

ROOT="${ROOT:-/root/grpo}"
export ROOT
export OPENR1_ROOT="${OPENR1_ROOT:-$ROOT/open-r1}"
export CFG="$ROOT/stage3/openr1_stage3_grpo_server_2x5090.yaml"

export RESUME_FROM_CHECKPOINT="/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-24244"

echo "=========================================="
echo "  Stage3 Training - Code + Math"
echo "=========================================="
echo ""
echo "Config: $CFG"
echo "Resume from: $RESUME_FROM_CHECKPOINT"
echo ""
echo "Training Plan:"
echo "  - Data: 51,888 samples (71% code, 29% math)"
echo "  - Duration: 0.5 epoch (~6,486 steps)"
echo "  - Checkpoints: Every 500 steps"
echo "  - max_completion_length: 1024 (code needs longer output)"
echo ""
echo "Key Parameters:"
echo "  - Learning rate: 1e-5"
echo "  - KL penalty (beta): 0.08"
echo "  - Accuracy weight: 1.3 (math correctness)"
echo "  - Format weight: 0.9 (already high at 91%)"
echo "  - Tag count weight: 0.2"
echo ""
echo "Monitoring:"
echo "  - MBPP pass@1 target: >= 65%"
echo "  - HumanEval pass@1 target: >= 60%"
echo "  - GSM8K guard: >= 50%"
echo "  - MATH guard: >= 33%"
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
