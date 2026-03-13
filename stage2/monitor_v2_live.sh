#!/usr/bin/env bash
# Live monitor for Stage2 v2 training (steps 22500 -> 24244)

BASE="/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2"

# Find the latest trainer_state.json (could be in output root or latest checkpoint)
STATE=""
for candidate in $(ls -dt "$BASE"/checkpoint-*/trainer_state.json "$BASE"/trainer_state.json 2>/dev/null); do
    STATE="$candidate"
    break
done

if [ -z "$STATE" ]; then
    echo "Waiting for training to start..."
    exit 0
fi

python3 << PYTHON
import json, sys, os, glob

base_dir = "$BASE"
# Find the newest trainer_state.json
candidates = glob.glob(os.path.join(base_dir, "checkpoint-*/trainer_state.json")) + \
             [os.path.join(base_dir, "trainer_state.json")]
candidates = [c for c in candidates if os.path.exists(c)]
state_file = max(candidates, key=os.path.getmtime) if candidates else None

if not state_file:
    print("Waiting for training to start...")
    sys.exit(0)
try:
    with open(state_file) as f:
        state = json.load(f)
except:
    print("Cannot read trainer_state.json")
    sys.exit(1)

logs = state.get('log_history', [])
global_step = state.get('global_step', 0)
max_steps = state.get('max_steps', 24244)

# Only show steps >= 22500 (stage2 v2 new training)
new_logs = [l for l in logs if l.get('step', 0) >= 22500]

print("=" * 65)
print("  Stage2 v2 Live Monitor")
print(f"  Progress: Step {global_step} / {max_steps} ({max(0, global_step-22500)}/{max_steps-22500} new steps)")
print("=" * 65)

if not new_logs:
    print("\nNo new training logs yet (fast-forwarding through old steps)...")
    sys.exit(0)

# Show recent training metrics
print(f"\n{'Step':>8} {'Loss':>10} {'Accuracy':>10} {'Format':>10} {'TagCount':>10} {'KL':>8} {'Reward':>8}")
print("-" * 65)

for log in new_logs:  # All entries
    step = log.get('step', '')
    loss = log.get('loss', '')
    acc = log.get('rewards/accuracy_reward/mean', '')
    fmt = log.get('rewards/format_reward/mean', '')
    tag = log.get('rewards/tag_count_reward/mean', '')
    kl = log.get('kl', '')
    reward = log.get('reward', '')

    if loss == '' and 'train_loss' in log:
        # Summary log entry
        continue

    row = f"{step:>8}"
    row += f"  {loss:>9.4f}" if isinstance(loss, (int, float)) else f"  {'':>9}"
    row += f"  {acc:>9.1%}" if isinstance(acc, (int, float)) else f"  {'':>9}"
    row += f"  {fmt:>9.1%}" if isinstance(fmt, (int, float)) else f"  {'':>9}"
    row += f"  {tag:>9.1%}" if isinstance(tag, (int, float)) else f"  {'':>9}"
    row += f"  {kl:>7.3f}" if isinstance(kl, (int, float)) else f"  {'':>7}"
    row += f"  {reward:>7.3f}" if isinstance(reward, (int, float)) else f"  {'':>7}"
    print(row)

# Show latest checkpoint
import os
ckpt_dir = "/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2"
ckpts = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")])
if ckpts:
    print(f"\nCheckpoints: {', '.join(ckpts)}")
PYTHON
