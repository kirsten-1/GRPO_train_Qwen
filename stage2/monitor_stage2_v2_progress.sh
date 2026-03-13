#!/usr/bin/env bash
# Monitor Stage2 v2 training progress by checking reward metrics

TRAINER_STATE="/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/trainer_state.json"

if [ ! -f "$TRAINER_STATE" ]; then
    echo "Training not started yet. Waiting for $TRAINER_STATE..."
    exit 1
fi

python3 << 'PYTHON'
import json
import sys

state_file = "/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/trainer_state.json"

try:
    with open(state_file) as f:
        state = json.load(f)
except:
    print("Cannot read trainer_state.json")
    sys.exit(1)

logs = state.get('log_history', [])
if not logs:
    print("No training logs yet")
    sys.exit(0)

# Get recent metrics
recent = logs[-10:] if len(logs) >= 10 else logs

print("=== Stage2 v2 Training Progress ===\n")
print(f"Total log entries: {len(logs)}")

# Calculate averages for recent steps
acc_rewards = []
fmt_rewards = []
steps = []

for entry in recent:
    step = entry.get('step')
    acc = entry.get('rewards/accuracy_reward/mean')
    fmt = entry.get('rewards/format_reward/mean')
    
    if step and acc is not None and fmt is not None:
        steps.append(step)
        acc_rewards.append(acc)
        fmt_rewards.append(fmt)

if acc_rewards:
    avg_acc = sum(acc_rewards) / len(acc_rewards)
    avg_fmt = sum(fmt_rewards) / len(fmt_rewards)
    
    print(f"\nRecent 10 steps ({steps[0]} - {steps[-1]}):")
    print(f"  Accuracy reward: {avg_acc:.4f} ({avg_acc*100:.1f}%)")
    print(f"  Format reward:   {avg_fmt:.4f} ({avg_fmt*100:.1f}%)")
    print(f"  Usable rate:     {avg_acc*avg_fmt:.4f} ({avg_acc*avg_fmt*100:.1f}%)")
    
    # Compare with Stage2 v1 baseline
    print(f"\nComparison with Stage2 v1:")
    print(f"  Accuracy: {avg_acc*100:.1f}% vs 55.1% (v1)")
    print(f"  Format:   {avg_fmt*100:.1f}% vs 44.1% (v1)")
    print(f"  Usable:   {avg_acc*avg_fmt*100:.1f}% vs 24.3% (v1)")
    
    # Show last entry details
    last = logs[-1]
    print(f"\nLatest step {last.get('step')}:")
    for k, v in last.items():
        if 'reward' in k.lower():
            print(f"  {k}: {v}")

PYTHON
