# GRPO 数据预处理

## 📁 文件说明

- `preprocess_datasets.py` - 主预处理脚本，格式化所有数据集
- `create_training_splits.py` - 创建三阶段训练集
- `validate_processed_data.py` - 验证数据质量
- `check_setup.py` - 环境检查
- `preview_format.py` - 数据格式预览

## 🚀 使用流程

```bash
cd /root/grpo/preprocess

# 1. 检查环境
python check_setup.py

# 2. 预处理数据
python preprocess_datasets.py

# 3. 创建训练集
python create_training_splits.py

# 4. 验证数据
python validate_processed_data.py
```

## 📊 训练策略（课程学习）

### Stage 1: 简单数学
- 数据: `stage1_simple_math_train.json`
- 内容: GSM8K + Ape210K (~208K 条)
- 训练: 2-3 epochs, lr=1e-5~5e-5

### Stage 2: 困难数学
- 数据: `stage2_hard_math_train.json`
- 内容: MATH×2 + CMATH (~4K 条)
- 训练: 2-3 epochs, lr=1e-5~5e-5

### Stage 3: 代码生成
- 数据: `stage3_code_train.json`
- 内容: MBPP×8 + APPS×4 + 15%数学回放 (~52K 条)
- 训练: 3-5 epochs, lr=5e-6~2e-5
- 💡 提示: MBPP 倍数已降至 ×8，如仍过拟合可降至 ×6

## ✨ 关键特性

- ✅ 保留 GSM8K 原始推理过程（解析 `####` 分隔符）
- ✅ 保留 MATH 原始 solution（提取 `\boxed{}` 答案）
- ✅ 课程学习：简单数学→困难数学→代码
- ✅ 数学经验回放（防止灾难性遗忘）
- ✅ 代码数据过采样（MBPP×8, APPS×4）
- ✅ MATH 上采样 ×2（增强困难数学能力）
- ✅ System Prompt 一致性验证

## 📂 输出目录

- 预处理数据: `/root/autodl-tmp/processed_datasets/`
- 训练数据: `/root/autodl-tmp/training_data/`

## 📝 数据集说明

### 数学数据集
- **Ape210K**: 中文小学数学 (~200K 条)
- **GSM8K**: 英文小学数学应用题 (~7.5K 条)
- **MATH**: 高中竞赛数学-代数 (~1.7K 条)
- **CMATH**: 中文高中数学 (~600 条)

### 代码数据集
- **MBPP**: Python 基础编程 (~120 条)
- **APPS**: Python 进阶编程 (~5K 条)
- **HumanEval**: Python 标准 benchmark (~164 条，仅测试)

