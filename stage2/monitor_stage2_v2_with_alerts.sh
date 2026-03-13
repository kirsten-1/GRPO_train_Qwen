#!/usr/bin/env bash
# Enhanced monitoring with decision alerts for Stage2 v2

TRAINER_STATE="/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/trainer_state.json"

if [ ! -f "$TRAINER_STATE" ]; then
    echo "⏳ Training not started yet. Waiting for $TRAINER_STATE..."
    exit 1
fi

python3 << 'PYTHON'
import json
import sys
from collections import defaultdict

state_file = "/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/trainer_state.json"

try:
    with open(state_file) as f:
        state = json.load(f)
except:
    print("❌ Cannot read trainer_state.json")
    sys.exit(1)

logs = state.get('log_history', [])
if not logs:
    print("⏳ No training logs yet")
    sys.exit(0)

print("=" * 60)
print("  Stage2 v2 Training Monitor - Real-time Analysis")
print("=" * 60)
print()

# Get current progress
last_entry = logs[-1]
current_step = last_entry.get('step', 0)
total_steps = 1744  # 0.5 epoch
progress = (current_step / total_steps) * 100 if total_steps > 0 else 0

print(f"📊 Progress: Step {current_step}/{total_steps} ({progress:.1f}%)")
print()

# Collect metrics by checkpoint intervals
checkpoints = defaultdict(list)
for entry in logs:
    step = entry.get('step', 0)
    acc = entry.get('rewards/accuracy_reward/mean')
    fmt = entry.get('rewards/format_reward/mean')
    
    if acc is not None and fmt is not None:
        # Group by 250-step intervals
        ckpt = (step // 250) * 250
        checkpoints[ckpt].append({'step': step, 'acc': acc, 'fmt': fmt})

# Calculate averages for each checkpoint interval
print("📈 Checkpoint Progress:")
print("-" * 60)
print(f"{'Checkpoint':<12} {'Accuracy':<12} {'Format':<12} {'Usable':<12} {'Status'}")
print("-" * 60)

# Baselines from Stage2 v1
baseline_acc = 0.5507
baseline_fmt = 0.4405
baseline_usable = 0.2426

decision_alerts = []

for ckpt in sorted(checkpoints.keys()):
    entries = checkpoints[ckpt]
    avg_acc = sum(e['acc'] for e in entries) / len(entries)
    avg_fmt = sum(e['fmt'] for e in entries) / len(entries)
    avg_usable = avg_acc * avg_fmt
    
    # Determine status
    status = "📊"
    if avg_fmt >= 0.60 and avg_acc >= 0.55:
        status = "✅ Good"
    elif avg_fmt >= 0.50 and avg_acc >= 0.50:
        status = "⚠️  Watch"
    elif avg_fmt < 0.45 or avg_acc < 0.45:
        status = "❌ Poor"
    
    print(f"Step {ckpt:<7} {avg_acc:>6.1%} ({avg_acc-baseline_acc:+.1%})  "
          f"{avg_fmt:>6.1%} ({avg_fmt-baseline_fmt:+.1%})  "
          f"{avg_usable:>6.1%} ({avg_usable-baseline_usable:+.1%})  {status}")
    
    # Decision alerts
    if ckpt == 250:
        if avg_fmt < 0.45:
            decision_alerts.append(f"🚨 ALERT at step 250: Format reward {avg_fmt:.1%} < 45% → Consider stopping")
        elif avg_fmt < 0.50:
            decision_alerts.append(f"⚠️  WARNING at step 250: Format reward {avg_fmt:.1%} in 45-50% range → Monitor closely")
        else:
            decision_alerts.append(f"✅ GOOD at step 250: Format reward {avg_fmt:.1%} ≥ 50% → Continue training")
    
    if ckpt == 500 and avg_acc < 0.45:
        decision_alerts.append(f"🚨 ALERT at step 500: Accuracy reward {avg_acc:.1%} < 45% → Consider stopping")

print("-" * 60)
print()

# Show recent trend (last 20 entries)
recent = logs[-20:] if len(logs) >= 20 else logs
recent_acc = [e.get('rewards/accuracy_reward/mean') for e in recent if e.get('rewards/accuracy_reward/mean') is not None]
recent_fmt = [e.get('rewards/format_reward/mean') for e in recent if e.get('rewards/format_reward/mean') is not None]

if recent_acc and recent_fmt:
    avg_recent_acc = sum(recent_acc) / len(recent_acc)
    avg_recent_fmt = sum(recent_fmt) / len(recent_fmt)
    avg_recent_usable = avg_recent_acc * avg_recent_fmt
    
    print("📊 Recent Trend (last 20 steps):")
    print(f"  Accuracy:  {avg_recent_acc:.1%} (baseline: {baseline_acc:.1%}, Δ{avg_recent_acc-baseline_acc:+.1%})")
    print(f"  Format:    {avg_recent_fmt:.1%} (baseline: {baseline_fmt:.1%}, Δ{avg_recent_fmt-baseline_fmt:+.1%})")
    print(f"  Usable:    {avg_recent_usable:.1%} (baseline: {baseline_usable:.1%}, Δ{avg_recent_usable-baseline_usable:+.1%})")
    print()

# Show decision alerts
if decision_alerts:
    print("🔔 Decision Alerts:")
    print("-" * 60)
    for alert in decision_alerts:
        print(f"  {alert}")
    print()

# Show latest metrics
print("📋 Latest Metrics (step {})".format(current_step))
print("-" * 60)
for k, v in sorted(last_entry.items()):
    if 'reward' in k.lower() and isinstance(v, (int, float)):
        print(f"  {k:<40} {v:.4f}")

print()
print("=" * 60)

PYTHON
