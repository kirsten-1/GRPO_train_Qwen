# Stage2 v2 实验计划

**创建时间**: 2026-03-11
**目标**: 通过优化超参数和reward权重，修复Stage2 v1的训练失败问题

---

## 问题诊断

Stage2 v1训练失败的根本原因：

1. **格式错误率高达55.95%** - 模型生成的答案格式不正确，无法被正确解析
2. **准确率仅55.07%** - 即使格式正确，答案也经常错误
3. **综合可用率仅24.26%** - 大部分训练样本质量极差
4. **学习率过低(5e-6)** - 在高噪声环境下更新太慢
5. **KL惩罚过高(0.10)** - 过度约束限制了模型探索空间

**结果**: MATH从38.0%下降到34.5-35.5%，GSM8K从54.5%暴跌到49.0-52.0%

---

## 优化方案

### 核心修改

| 参数 | Stage2 v1 | Stage2 v2 | 理由 |
|------|-----------|-----------|------|
| **learning_rate** | 5e-6 | **1e-5** | 恢复Stage1水平，加快学习 |
| **beta (KL惩罚)** | 0.10 | **0.08** | 降低约束，允许更大探索空间 |
| **accuracy权重** | 1.3 | **1.2** | 降低难题惩罚，鼓励探索 |
| **format权重** | 0.9 | **1.5** | **关键修改**：优先解决格式问题 |
| **num_train_epochs** | 1.0 | **0.5** | 快速验证，降低风险 |
| **save_steps** | 500 | **250** | 密集评估，及时发现问题 |
| **起始checkpoint** | 22500 | **22500** | 已验证最优（MATH 38.0%） |

### 为什么从checkpoint-22500开始？

✅ **支持理由**：
1. 快速评估证明它是最佳checkpoint（MATH 38.0%, GSM8K 54.5%）
2. Stage1仅训练1 epoch，过拟合风险极低
3. 在未训练的MATH上表现最好，说明泛化能力强

❌ **反对从22000开始的理由**：
1. 缺少评估数据，不知道真实表现
2. 逻辑矛盾：如果22500过拟合，为什么在MATH上表现最好？

---

## 训练计划

### 基本参数
- **训练步数**: 1,744步（0.5 epoch）
- **预计时长**: 约2小时
- **Checkpoint间隔**: 每250步（共7个checkpoint）
- **评估点**: 步数250, 500, 750, 1000, 1250, 1500, 1744

### 监控指标

**主要指标**（训练过程中实时监控）：
- `rewards/accuracy_reward/mean` - 准确率奖励
- `rewards/format_reward/mean` - 格式奖励
- `frac_reward_zero_std` - 零方差奖励比例

**评估指标**（每个checkpoint快速评估）：
- MATH准确率（200样本）
- GSM8K准确率（200样本）
- MBPP pass@1（128样本）
- HumanEval pass@1（82样本）

---

## 决策流程

### 关键决策点

#### 第250步（训练进度14%）
- ✅ **继续训练**: format_reward > 50% 且 accuracy_reward > 50%
- ⚠️ **密切观察**: format_reward 45-50%，继续到500步再判断
- ❌ **考虑停止**: format_reward < 45%，说明reward权重调整无效

#### 第500步（训练进度29%）
- ✅ **继续训练**: format_reward持续提升，accuracy_reward保持
- ⚠️ **警戒**: GSM8K下降超2%（低于52.5%），可能灾难性遗忘
- ❌ **停止训练**: accuracy_reward < 45% 或 format_reward无改善

#### 第1000步（训练进度57%）
- ✅ **继续到完成**: 指标持续改善
- ⚠️ **评估是否继续**: 指标平稳但未达目标
- ❌ **停止**: 指标恶化或GSM8K下降超3%

### 成功标准（0.5 epoch结束）

**必须达成**：
- Format reward ≥ 60%（从44.05%提升）
- Accuracy reward ≥ 60%（从55.07%提升）
- GSM8K ≥ 53.5%（允许下降1%）

**期望达成**：
- MATH ≥ 40%（从38.0%提升2%）
- Usable rate ≥ 35%（从24.26%提升）

**如果成功** → 继续训练到完整1 epoch
**如果失败** → 考虑更根本的策略（见下文）

---

## 备选方案

如果Stage2 v2仍然失败，考虑以下策略：

### 方案A: Curriculum Learning
1. **阶段1**: SFT训练MATH简单题（difficulty 1-2）
2. **阶段2**: GRPO训练MATH中等题（difficulty 3-4）
3. **阶段3**: GRPO训练MATH困难题（difficulty 5）

### 方案B: 混合训练
- MATH + GSM8K混合（比例3:1）
- 避免纯MATH太难，保持GSM8K能力

### 方案C: 更强Base Model
- 当前Qwen2.5-3B可能对MATH难度天花板太低
- 考虑升级到7B或14B模型

### 方案D: 接受现状
- checkpoint-22500（MATH 38.0%）已经是不错的结果
- 专注于Stage3代码能力提升

---

## 执行步骤

### 1. 启动训练
```bash
cd /root/grpo/stage2
./run_stage2_v2_from_ckpt22500.sh
```

### 2. 实时监控
```bash
# 每10分钟检查一次
watch -n 600 /root/grpo/stage2/monitor_stage2_v2_with_alerts.sh

# 或手动检查
./monitor_stage2_v2_with_alerts.sh
```

### 3. Checkpoint评估
```bash
# 在每个checkpoint生成后运行
./quick_eval_checkpoint.sh /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-22750
./quick_eval_checkpoint.sh /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-23000
# ... 依此类推
```

### 4. 分析结果
- 对比各checkpoint的训练指标和评估指标
- 绘制format_reward和accuracy_reward曲线
- 判断是否继续训练到1 epoch

---

## 风险提示

1. **格式问题可能是模型能力问题** - 如果250步后format_reward仍无改善，说明不是reward权重的问题，而是模型基础能力不足
2. **MATH可能对3B模型太难** - 即使优化超参数，3B模型可能无法突破40%的天花板
3. **灾难性遗忘风险** - 需要密切监控GSM8K，如果下降超3%应立即停止

---

## 文件清单

**配置文件**：
- `/root/grpo/stage2/openr1_stage2_grpo_server_2x5090_v2.yaml` - 优化后的训练配置

**脚本文件**：
- `/root/grpo/stage2/run_stage2_v2_from_ckpt22500.sh` - 训练启动脚本
- `/root/grpo/stage2/monitor_stage2_v2_with_alerts.sh` - 增强监控脚本
- `/root/grpo/stage2/quick_eval_checkpoint.sh` - 快速评估脚本

**输出目录**：
- `/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/` - 训练输出
- `/root/autodl-tmp/stage2_eval_runs/stage2_v2_*/` - 评估结果

---

**实验负责人**: 用户 + Claude Code
**预计完成时间**: 2026-03-11（训练2小时 + 评估7小时）
