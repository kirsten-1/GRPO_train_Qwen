#!/usr/bin/env bash
set -euo pipefail

# Smoke test: run 3 steps from checkpoint-22500 to verify everything works
# Expected: starts training, logs 3 steps, then exits cleanly

ROOT="${ROOT:-/root/grpo}"
export ROOT
export OPENR1_ROOT="${OPENR1_ROOT:-$ROOT/open-r1}"
export CFG="$ROOT/stage2/openr1_stage2_grpo_server_2x5090_v2.yaml"
export RESUME_FROM_CHECKPOINT="/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090/checkpoint-22500"

echo "============================================"
echo "  Stage2 v2 SMOKE TEST (3 steps only)"
echo "============================================"
echo ""
echo "Config: $CFG"
echo "Resume from: $RESUME_FROM_CHECKPOINT"
echo "Will run only 3 steps then stop."
echo ""

# Override max_steps to just 3 beyond checkpoint
exec bash "$ROOT/stage1/run_stage1_openr1_server.sh" \
  --max_steps 22503 \
  --save_steps 999999 \
  --logging_steps 1 \
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
