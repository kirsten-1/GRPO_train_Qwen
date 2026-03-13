#!/usr/bin/env bash
# Stage3 monitor: terminal shows recent 2 checkpoints, full history saved to MD file

BASE="/root/autodl-tmp/stage3_grpo_runs/openr1_stage3_server_2x5090"
MD_FILE="/root/grpo/stage3/stage3_training_log.md"

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
import json, sys, os, glob, math
from datetime import datetime

base_dir = "$BASE"
md_file = "$MD_FILE"

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
max_steps = state.get('max_steps', 30730)
start_step = 24244
save_steps = 500

new_logs = [l for l in logs if l.get('step', 0) >= start_step and 'loss' in l]

ckpts = sorted([d for d in os.listdir(base_dir) if d.startswith("checkpoint-")])
ckpt_steps = sorted([int(c.split("-")[1]) for c in ckpts])

def fmt_row(log):
    step = log.get('step', '')
    loss = log.get('loss', '')
    acc = log.get('rewards/accuracy_reward/mean', '')
    fmt = log.get('rewards/format_reward/mean', '')
    tag = log.get('rewards/tag_count_reward/mean', '')
    kl = log.get('kl', '')
    reward = log.get('reward', '')
    row = f"{step:>8}"
    row += f"  {loss:>9.4f}" if isinstance(loss, (int, float)) else f"  {'':>9}"
    if isinstance(acc, float) and math.isnan(acc):
        row += f"  {'--':>9}"
    elif isinstance(acc, (int, float)):
        row += f"  {acc:>9.1%}"
    else:
        row += f"  {'':>9}"
    row += f"  {fmt:>9.1%}" if isinstance(fmt, (int, float)) else f"  {'':>9}"
    row += f"  {tag:>9.1%}" if isinstance(tag, (int, float)) else f"  {'':>9}"
    row += f"  {kl:>7.3f}" if isinstance(kl, (int, float)) else f"  {'':>7}"
    row += f"  {reward:>7.3f}" if isinstance(reward, (int, float)) else f"  {'':>7}"
    return row

def fmt_row_md(log):
    step = log.get('step', '')
    loss = log.get('loss', '')
    acc = log.get('rewards/accuracy_reward/mean', '')
    fmt = log.get('rewards/format_reward/mean', '')
    tag = log.get('rewards/tag_count_reward/mean', '')
    kl = log.get('kl', '')
    reward = log.get('reward', '')
    s_loss = f"{loss:.4f}" if isinstance(loss, (int, float)) else ""
    if isinstance(acc, float) and math.isnan(acc):
        s_acc = "--"
    elif isinstance(acc, (int, float)):
        s_acc = f"{acc:.1%}"
    else:
        s_acc = ""
    s_fmt = f"{fmt:.1%}" if isinstance(fmt, (int, float)) else ""
    s_tag = f"{tag:.1%}" if isinstance(tag, (int, float)) else ""
    s_kl = f"{kl:.3f}" if isinstance(kl, (int, float)) else ""
    s_reward = f"{reward:.3f}" if isinstance(reward, (int, float)) else ""
    return f"| {step} | {s_loss} | {s_acc} | {s_fmt} | {s_tag} | {s_kl} | {s_reward} |"

header = f"{'Step':>8} {'Loss':>10} {'Accuracy':>10} {'Format':>10} {'TagCount':>10} {'KL':>8} {'Reward':>8}"
sep = "-" * 65

# --- Determine cutoff: show last 2 checkpoints worth of data ---
if len(ckpt_steps) >= 2:
    cutoff_step = ckpt_steps[-2]
elif len(ckpt_steps) == 1:
    cutoff_step = ckpt_steps[0]
else:
    cutoff_step = max(start_step, global_step - 2 * save_steps)

recent_logs = [l for l in new_logs if l.get('step', 0) >= cutoff_step]

progress_pct = (global_step - start_step) / (max_steps - start_step) * 100 if max_steps > start_step else 0

# === Terminal output: recent only ===
print("=" * 65)
print("  Stage3 Training Monitor (Code + Math)")
print(f"  Progress: Step {global_step} / {max_steps}  ({progress_pct:.1f}%)")
print(f"  New steps: {max(0, global_step-start_step)} / {max_steps-start_step}")
print(f"  Source: {os.path.basename(os.path.dirname(state_file))}")
print(f"  Showing: step {cutoff_step}+ (last 2 checkpoints)")
print("=" * 65)

if recent_logs:
    print(f"\n{header}")
    print(sep)
    for log in recent_logs:
        print(fmt_row(log))
else:
    print("\nNo logs in this range yet...")

if ckpts:
    print(f"\nCheckpoints: {', '.join(ckpts)}")
print(f"\nFull log saved to: {md_file}")

# === MD file: full history ===
with open(md_file, 'w') as f:
    f.write(f"# Stage3 Training Log\n\n")
    f.write(f"**Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    f.write(f"**Progress**: Step {global_step} / {max_steps} ({progress_pct:.1f}%)  \n")
    f.write(f"**New steps**: {max(0, global_step-start_step)} / {max_steps-start_step}  \n")
    f.write(f"**Checkpoints**: {', '.join(ckpts)}  \n\n")
    f.write(f"## Training Metrics\n\n")
    f.write(f"| Step | Loss | Accuracy | Format | TagCount | KL | Reward |\n")
    f.write(f"|------|------|----------|--------|----------|----|--------|\n")
    for log in new_logs:
        f.write(fmt_row_md(log) + "\n")

PYTHON
