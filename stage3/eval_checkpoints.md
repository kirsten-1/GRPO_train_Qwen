# Stage3 Checkpoint 评估命令

训练完成：step 24244 → 30730
现存 checkpoints: 27500, 28000, 28500, 29000, 29500, 30000, 30500, 30730
（早期 checkpoint 24500~27000 已被 save_total_limit=8 清理）

评估数据集：smoke suite

| 数据集 | 类型 | 样本数 | 总集大小 |
|--------|------|--------|----------|
| MATH | 数学 | 128 | 1,187 |
| CMATH | 数学 | 128 | 1,098 |
| GSM8K | 数学 | 128 | 1,319 |
| APE210K | 数学 | 128 | 5,000 |
| MBPP | 代码 | 32 | 257 |
| HumanEval | 代码 | 32 | 164 |
| **合计** | | **576** | |

## 评估命令（GPU 0/1 交替，每批 2 个并行）

### 第 1 批

```bash
# 终端 A - GPU 0
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage3_grpo_runs/openr1_stage3_server_2x5090/checkpoint-27500 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage3_ckpt27500

# 终端 B - GPU 1
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage3_grpo_runs/openr1_stage3_server_2x5090/checkpoint-28000 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage3_ckpt28000
```

### 第 2 批

```bash
# 终端 A - GPU 0
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage3_grpo_runs/openr1_stage3_server_2x5090/checkpoint-28500 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage3_ckpt28500

# 终端 B - GPU 1
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage3_grpo_runs/openr1_stage3_server_2x5090/checkpoint-29000 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage3_ckpt29000
```

### 第 3 批

```bash
# 终端 A - GPU 0
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage3_grpo_runs/openr1_stage3_server_2x5090/checkpoint-29500 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage3_ckpt29500

# 终端 B - GPU 1
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage3_grpo_runs/openr1_stage3_server_2x5090/checkpoint-30000 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage3_ckpt30000
```

### 第 4 批

```bash
# 终端 A - GPU 0
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage3_grpo_runs/openr1_stage3_server_2x5090/checkpoint-30500 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage3_ckpt30500

# 终端 B - GPU 1
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage3_grpo_runs/openr1_stage3_server_2x5090/checkpoint-30730 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage3_ckpt30730
```

## 查看所有结果

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('/root/autodl-tmp/stage2_eval_runs/stage3_ckpt*/eval_results.json')):
    with open(f) as fh:
        m = json.load(fh).get('metrics', {})
    name = f.split('/')[-2]
    math = m.get('math_accuracy', 0)
    gsm = m.get('gsm8k_accuracy', 0)
    cmath = m.get('cmath_accuracy', 0)
    fmt = m.get('math_format_correct_rate', 0)
    mbpp = m.get('mbpp_pass@1', m.get('mbpp_pass1', 0))
    he = m.get('humaneval_pass@1', m.get('humaneval_pass1', 0))
    print(f'{name:30s}  MATH={math:.1%}  GSM8K={gsm:.1%}  CMATH={cmath:.1%}  MBPP={mbpp:.1%}  HE={he:.1%}  Fmt={fmt:.1%}')
"
```

## Stage2 基线评估（ckpt-24244，按数据集拆分）

### 第 1 批：数学

```bash
# 终端 A - GPU 0 - MATH+CMATH
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-24244 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --gsm8k-samples 0 --ape210k-samples 0 --mbpp-samples 0 --humaneval-samples 0 \
  --run-name stage2_ckpt24244_math_cmath

# 终端 B - GPU 1 - GSM8K+APE210K
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-24244 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --math-samples 0 --cmath-samples 0 --mbpp-samples 0 --humaneval-samples 0 \
  --run-name stage2_ckpt24244_gsm8k_ape
```

### 第 2 批：代码

```bash
# 终端 A - GPU 0 - MBPP
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-24244 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --math-samples 0 --cmath-samples 0 --gsm8k-samples 0 --ape210k-samples 0 --humaneval-samples 0 \
  --run-name stage2_ckpt24244_mbpp

# 终端 B - GPU 1 - HumanEval
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-24244 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --math-samples 0 --cmath-samples 0 --gsm8k-samples 0 --ape210k-samples 0 --mbpp-samples 0 \
  --run-name stage2_ckpt24244_humaneval
```

## 监控指标警戒线

| 指标 | 基线 (ckpt-24244) | 目标 | 警戒线 |
|------|-------------------|------|--------|
| MBPP pass@1 | 62.5% | >= 65% | < 55% |
| HumanEval pass@1 | 53.1% | >= 60% | < 50% |
| GSM8K | 55.5% | 维持 | < 50% |
| MATH | 35.9% | 维持 | < 33% |
