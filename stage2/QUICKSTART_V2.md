# Stage2 v2 快速启动指南

## 一键启动

```bash
cd /root/grpo/stage2
./run_stage2_v2_from_ckpt22500.sh
```

## 监控训练

**实时监控**（推荐）：
```bash
watch -n 600 /root/grpo/stage2/monitor_stage2_v2_with_alerts.sh
```

**手动检查**：
```bash
./monitor_stage2_v2_with_alerts.sh
```

## 关键决策点

### 第250步（约30分钟后）
检查format_reward：
- ✅ **> 50%** → 继续训练
- ⚠️ **45-50%** → 观察到500步
- ❌ **< 45%** → 考虑停止

### 第500步（约1小时后）
检查accuracy_reward和GSM8K：
- ✅ **accuracy > 50%, GSM8K > 52.5%** → 继续
- ❌ **accuracy < 45% 或 GSM8K < 52%** → 停止

## 评估Checkpoint

```bash
# 等待checkpoint生成后
./quick_eval_checkpoint.sh /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-22750
```

## 预期结果

**成功标准**（0.5 epoch结束）：
- Format reward: 44% → 60%+
- Accuracy reward: 55% → 60%+
- MATH: 38% → 40%+
- GSM8K: 54.5% → 53.5%+（允许-1%）

## 如果失败

参考 `STAGE2_V2_EXPERIMENT_PLAN.md` 中的备选方案。
