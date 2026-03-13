# Stage2 v2 四个 Checkpoint 评估命令

评估数据集：smoke suite（MATH 128, CMATH 128, GSM8K 128, APE210K 128, MBPP 32, HumanEval 32）
预计时间：每轮 15-20 分钟，两轮共 30-40 分钟

## 第一轮（2 个终端并行）

终端 1（GPU 0）：
```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-23000 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage2v2_ckpt23000
```

终端 2（GPU 1）：
```bash
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-23500 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage2v2_ckpt23500
```

## 第二轮（等第一轮跑完后）

终端 1（GPU 0）：
```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-24000 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage2v2_ckpt24000
```

终端 2（GPU 1）：
```bash
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090_v2/checkpoint-24244 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite smoke --run-name stage2v2_ckpt24244
```

## 查看结果

```bash
# 列出所有评估结果
ls /root/autodl-tmp/stage2_eval_runs/stage2v2_ckpt*/eval_results.json

# 对比各 checkpoint 的核心指标
python3 -c "
import json, glob
for f in sorted(glob.glob('/root/autodl-tmp/stage2_eval_runs/stage2v2_ckpt*/eval_results.json')):
    with open(f) as fh:
        m = json.load(fh).get('metrics', {})
    name = f.split('/')[-2]
    math = m.get('math_accuracy', 0)
    gsm = m.get('gsm8k_accuracy', 0)
    cmath = m.get('cmath_accuracy', 0)
    fmt = m.get('math_format_correct_rate', 0)
    print(f'{name:30s}  MATH={math:.1%}  GSM8K={gsm:.1%}  CMATH={cmath:.1%}  Format={fmt:.1%}')
"
```
