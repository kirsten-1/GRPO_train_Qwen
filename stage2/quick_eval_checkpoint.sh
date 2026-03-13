#!/usr/bin/env bash
# Quick evaluation for Stage2 v2 checkpoints
# Usage: ./quick_eval_checkpoint.sh <checkpoint_path>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-22750"
    exit 1
fi

CHECKPOINT="$1"
CKPT_NAME=$(basename "$CHECKPOINT")

if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "  Quick Evaluation: $CKPT_NAME"
echo "=========================================="
echo ""
echo "Checkpoint: $CHECKPOINT"
echo "Evaluation: MATH (200), GSM8K (200), MBPP (128), HumanEval (82)"
echo "Expected duration: ~60 minutes"
echo ""

# Set environment variables for quick evaluation
export EVAL_MATH_SAMPLES=200
export EVAL_GSM8K_SAMPLES=200
export EVAL_MBPP_SAMPLES=128
export EVAL_HUMANEVAL_SAMPLES=82
export EVAL_CODE_PASS_K=1
export EVAL_TIMEOUT=5

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/root/autodl-tmp/stage2_eval_runs/stage2_v2_${CKPT_NAME}_quick_${TIMESTAMP}"

echo "Output: $OUTPUT_DIR"
echo ""
read -p "Press Enter to start evaluation, or Ctrl+C to cancel..."

# Run evaluation (you'll need to adapt this to your actual eval script)
cd /root/grpo/stage2
python3 /root/grpo/open-r1/scripts/eval_openr1.py \
    --model_path "$CHECKPOINT" \
    --base_model_path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
    --output_dir "$OUTPUT_DIR" \
    --suite retain \
    --math_samples $EVAL_MATH_SAMPLES \
    --gsm8k_samples $EVAL_GSM8K_SAMPLES \
    --mbpp_samples $EVAL_MBPP_SAMPLES \
    --humaneval_samples $EVAL_HUMANEVAL_SAMPLES \
    --code_pass_k $EVAL_CODE_PASS_K \
    --timeout $EVAL_TIMEOUT

echo ""
echo "✅ Evaluation complete!"
echo "Results: $OUTPUT_DIR/eval_results.json"
echo ""

# Show quick summary
if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
    python3 << PYTHON
import json
with open("$OUTPUT_DIR/eval_results.json") as f:
    data = json.load(f)
    m = data.get('metrics', {})
    print("Quick Summary:")
    print(f"  MATH:      {m.get('math_accuracy', 0):.1%}")
    print(f"  GSM8K:     {m.get('gsm8k_accuracy', 0):.1%}")
    print(f"  MBPP:      {m.get('mbpp_pass1', 0):.1%}")
    print(f"  HumanEval: {m.get('humaneval_pass1', 0):.1%}")
PYTHON
fi
