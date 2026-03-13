# Stage2 训练记忆文档

**创建时间**: 2026-03-11 04:49 UTC
**状态**: Stage2 v2准备就绪，等待启动训练
**目的**: 无卡模式→有卡模式切换时快速恢复上下文

---

## 当前状态总结

### Stage2 v1 训练结果（已完成，失败）

**训练配置**：
- 起始checkpoint: checkpoint-22000
- 训练数据: 3,488条纯MATH样本
- 训练步数: 3,488步（1 epoch）
- 学习率: 5e-6
- KL惩罚(beta): 0.10
- Reward权重: accuracy=1.3, format=0.9, tag=0.2

**训练指标（平均值）**：
- Accuracy reward: 55.07% → 44.93%答案错误
- Format reward: 44.05% → **55.95%格式错误**（最大问题）
- Tag count reward: 73.64%
- **综合可用率: 仅24.26%**（75.74%样本不可用）

**评估结果（5个checkpoints快速评估）**：

| Checkpoint | MATH | GSM8K | MBPP | HumanEval | 评价 |
|-----------|------|-------|------|-----------|------|
| 22500 (起点) | 38.0% | 54.5% | 53.9% | 42.7% | ✅ 最佳 |
| 23500 | 34.5% | 52.0% | 54.7% | 53.7% | ❌ 数学退步 |
| 24000 | 34.5% | 51.0% | 58.6% | 52.4% | ❌ 数学退步 |
| 24500 | 35.5% | 49.0% | 56.3% | 52.4% | 🚨 灾难性遗忘 |
| 25000 | 35.5% | 49.5% | 55.5% | 48.8% | 🚨 灾难性遗忘 |

**失败原因**：
1. 格式错误率高达55.95% - 模型生成答案格式不正确
2. 学习率5e-6太保守 - 在高噪声环境下学不到有效信号
3. KL惩罚0.10过高 - 过度约束限制探索空间
4. Format权重0.9太低 - 没有优先解决格式问题

**结论**: checkpoint-22500仍是最佳模型，Stage2 v1训练完全失败

---

## Stage2 v2 优化方案（准备启动）

### 核心修改

| 参数 | v1 | v2 | 改进幅度 | 理由 |
|------|----|----|---------|------|
| learning_rate | 5e-6 | **1e-5** | +100% | 恢复Stage1水平，加快学习 |
| beta (KL惩罚) | 0.10 | **0.08** | -20% | 降低约束，增大探索空间 |
| accuracy权重 | 1.3 | **1.2** | -7.7% | 降低难题惩罚 |
| format权重 | 0.9 | **1.5** | **+66.7%** | **关键修改**：优先解决格式问题 |
| num_train_epochs | 1.0 | **0.5** | - | 快速验证，降低风险 |
| save_steps | 500 | **250** | - | 密集评估，7个checkpoint |
| 起始checkpoint | 22500 | **22500** | - | 已验证最优 |

### 训练计划

- **总步数**: 1,744步（0.5 epoch）
- **预计时长**: 约2小时
- **Checkpoint间隔**: 每250步
- **生成checkpoints**: 22750, 23000, 23250, 23500, 23750, 24000, 24244

### 成功标准（0.5 epoch结束）

**必须达成**：
- Format reward: 44% → **≥60%**（提升15.95pp）
- Accuracy reward: 55% → **≥60%**（提升4.93pp）
- GSM8K: 54.5% → **≥53.5%**（允许下降1%）

**期望达成**：
- MATH: 38.0% → **≥40%**（提升2%）
- Usable rate: 24.26% → **≥35%**（提升10.74pp）

---

## 关键决策点

### 第250步（约30分钟后）- 关键决策点1

**检查指标**: `rewards/format_reward/mean`

- ✅ **format_reward > 50%** → 继续训练，策略有效
- ⚠️ **format_reward 45-50%** → 观察到500步再判断
- ❌ **format_reward < 45%** → 考虑停止，reward权重调整无效

### 第500步（约1小时后）- 关键决策点2

**检查指标**: `rewards/accuracy_reward/mean` 和 GSM8K评估结果

- ✅ **accuracy > 50% 且 GSM8K > 52.5%** → 继续训练
- ⚠️ **GSM8K下降2-3%** → 警戒，密切监控
- ❌ **accuracy < 45% 或 GSM8K < 52%** → 停止训练

### 第1000步（约1.5小时后）- 中期评估

**检查指标**: MATH和GSM8K快速评估

- ✅ **MATH ≥ 38.5%, GSM8K ≥ 53%** → 继续到完成
- ⚠️ **MATH无提升但GSM8K稳定** → 评估是否继续
- ❌ **MATH下降或GSM8K < 52%** → 停止训练

### 第1744步（约2小时后）- 最终评估

**决策**：
- ✅ **达成成功标准** → 继续训练到完整1 epoch
- ⚠️ **部分达成** → 分析原因，调整策略
- ❌ **未达成** → 停止，考虑备选方案

---

## 如何分析训练日志

### 关键指标位置

训练日志中需要关注的指标（每10步输出一次）：

```
Step XXXX:
  reward: X.XXXX                              # 总reward
  reward_std: X.XXXX                          # reward标准差
  rewards/accuracy_reward/mean: X.XXXX        # ← 准确率奖励（目标≥0.60）
  rewards/accuracy_reward/std: X.XXXX
  rewards/format_reward/mean: X.XXXX          # ← 格式奖励（目标≥0.60）
  rewards/format_reward/std: X.XXXX
  rewards/tag_count_reward/mean: X.XXXX       # ← 标签奖励
  rewards/tag_count_reward/std: X.XXXX
  frac_reward_zero_std: X.XXXX                # 零方差比例
  loss: X.XXXX                                # 训练loss
  learning_rate: X.XXXXe-XX                   # 当前学习率
```

### 健康训练的信号

**好的迹象**：
- `rewards/format_reward/mean` 在前250步内从0.44提升到0.50+
- `rewards/accuracy_reward/mean` 保持在0.50-0.60之间
- `reward_std` 保持在0.3-0.6之间（不要太低或太高）
- `loss` 逐步下降但不要降到0（过拟合）
- `frac_reward_zero_std` < 0.3（不要太多样本reward相同）

**坏的迹象**：
- `rewards/format_reward/mean` 在250步后仍<0.45
- `rewards/accuracy_reward/mean` 持续下降到<0.45
- `reward_std` 接近0（所有样本reward相同，学不到东西）
- `loss` 快速降到接近0（过拟合）
- `frac_reward_zero_std` > 0.5（大量样本无区分度）

### 对比基准（Stage2 v1平均值）

用这些数值作为对比基准：
- Accuracy reward baseline: 0.5507
- Format reward baseline: 0.4405
- Tag count reward baseline: 0.7364

如果v2的指标持续低于这些基准，说明优化无效。

---

## 监控和评估命令

### 实时监控训练进度

```bash
# 方法1：自动刷新（推荐）
watch -n 600 /root/grpo/stage2/monitor_stage2_v2_with_alerts.sh

# 方法2：手动检查
/root/grpo/stage2/monitor_stage2_v2_with_alerts.sh
```

监控脚本会显示：
- 当前进度（步数/总步数）
- 各checkpoint的平均指标
- 与baseline的对比
- 决策提醒（在关键步数时）

### 快速评估checkpoint

```bash
# 在checkpoint生成后运行
/root/grpo/stage2/quick_eval_checkpoint.sh \
  /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-22750
```

评估耗时约60分钟，会输出：
- MATH准确率（200样本）
- GSM8K准确率（200样本）
- MBPP pass@1（128样本）
- HumanEval pass@1（82样本）

### 查看原始训练日志

```bash
# 查看最新日志
tail -f /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/runs/*/events.out.tfevents.*

# 或查看trainer_state.json
python3 -c "
import json
with open('/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/trainer_state.json') as f:
    state = json.load(f)
    logs = state['log_history']
    # 显示最近10条
    for entry in logs[-10:]:
        print(f\"Step {entry.get('step')}: \", entry)
"
```

---

## 重要文件位置

### 配置和脚本

- **训练配置**: `/root/grpo/stage2/openr1_stage2_grpo_server_2x5090_v2.yaml`
- **启动脚本**: `/root/grpo/stage2/run_stage2_v2_from_ckpt22500.sh`
- **监控脚本**: `/root/grpo/stage2/monitor_stage2_v2_with_alerts.sh`
- **评估脚本**: `/root/grpo/stage2/quick_eval_checkpoint.sh`

### 数据和模型

- **训练数据**: `/root/autodl-tmp/training_data/stage2_hf/train.jsonl` (3,488条)
- **Base模型**: `/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct`
- **起始checkpoint**: `/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090/checkpoint-22500`

### 输出目录

- **训练输出**: `/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/`
  - checkpoints: `checkpoint-22750/`, `checkpoint-23000/`, ...
  - 训练状态: `trainer_state.json`
  - TensorBoard日志: `runs/`
- **评估结果**: `/root/autodl-tmp/stage2_eval_runs/stage2_v2_*/`

### 文档

- **实验计划**: `/root/grpo/stage2/STAGE2_V2_EXPERIMENT_PLAN.md`
- **快速启动**: `/root/grpo/stage2/QUICKSTART_V2.md`
- **配置对比**: `/root/grpo/stage2/CONFIG_COMPARISON.md`
- **v1评估分析**: `/root/grpo/stage2/STAGE2_EVALUATION_ANALYSIS.md`

---

## 启动训练

```bash
cd /root/grpo/stage2
./run_stage2_v2_from_ckpt22500.sh
```

脚本会显示配置摘要，按Enter确认后开始训练。

---

## 用户粘贴日志时的分析流程

当用户粘贴训练日志时，按以下流程分析：

### 1. 识别当前步数和进度

从日志中找到 `Step XXXX`，计算进度：
- 进度 = 当前步数 / 1744 × 100%
- 判断是否到达关键决策点（250, 500, 1000步）

### 2. 提取关键指标

从日志中提取：
- `rewards/accuracy_reward/mean`
- `rewards/format_reward/mean`
- `rewards/tag_count_reward/mean`
- `reward_std`
- `loss`

### 3. 对比基准

与Stage2 v1基准对比：
- Accuracy: 当前值 vs 0.5507
- Format: 当前值 vs 0.4405
- 计算改进幅度（正数=改善，负数=恶化）

### 4. 判断趋势

- 如果是前10-20步：正常波动，不用担心
- 如果是100-250步：观察format_reward是否上升趋势
- 如果是250步附近：关键决策点，判断是否达到50%
- 如果是500步附近：检查accuracy和GSM8K
- 如果是1000步以上：评估整体效果

### 5. 给出建议

根据指标和趋势，给出：
- ✅ 继续训练
- ⚠️ 密切观察
- ❌ 考虑停止
- 📊 建议运行checkpoint评估

### 6. 示例分析

**示例日志**：
```
Step 250:
  rewards/accuracy_reward/mean: 0.525
  rewards/format_reward/mean: 0.48
  reward_std: 0.45
```

**分析**：
- 进度: 250/1744 = 14.3%（关键决策点1）
- Accuracy: 0.525 vs 0.5507 baseline（-2.57pp，轻微下降）
- Format: 0.48 vs 0.4405 baseline（+3.95pp，有改善但未达50%目标）
- 判断: ⚠️ 观察 - format有改善但未达目标，继续到500步再判断

---

## 备选方案（如果v2失败）

### 方案A: Curriculum Learning
1. 阶段1: SFT训练MATH简单题（difficulty 1-2）
2. 阶段2: GRPO训练MATH中等题（difficulty 3-4）
3. 阶段3: GRPO训练MATH困难题（difficulty 5）

### 方案B: 混合训练
- MATH + GSM8K混合（比例3:1）
- 避免纯MATH太难

### 方案C: 更强Base Model
- 升级到Qwen2.5-7B或14B
- 3B可能对MATH天花板太低

### 方案D: 接受现状
- checkpoint-22500（MATH 38.0%）已经不错
- 专注Stage3代码能力

---

## 常见问题

**Q: 训练卡住不动怎么办？**
A: 检查vLLM server是否正常运行（GPU 1），查看端口8000是否监听

**Q: 格式错误率一直不降怎么办？**
A: 说明模型基础能力不足，考虑降低MATH难度或换更强模型

**Q: GSM8K下降超过3%怎么办？**
A: 立即停止训练，发生灾难性遗忘，回退到checkpoint-22500

**Q: 0.5 epoch后要不要继续到1 epoch？**
A: 只有达成成功标准才继续，否则浪费资源

**Q: 如何判断是否过拟合？**
A: 看loss是否接近0，reward_std是否接近0，frac_reward_zero_std是否>0.5

---

**最后更新**: 2026-03-11 04:49 UTC
**状态**: 准备就绪，等待有卡模式启动训练
