# Baseline Evaluation

脚本：`/root/grpo/baseline/run_full_baseline.py`

## 关键点
- 已硬编码本地模型路径：`/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct`
- 启用离线模式：`HF_DATASETS_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`
- 输出目录默认：`/root/autodl-tmp/baseline_results/`

## 先做无卡检查
```bash
python /root/grpo/baseline/run_full_baseline.py --dry-run
```

## 全量 baseline（建议有卡后运行）
```bash
python /root/grpo/baseline/run_full_baseline.py \
  --datasets gsm8k,ape210k,math,cmath,mbpp,humaneval,apps \
  --ape-max-samples 5000 \
  --code-num-samples 10 \
  --math-max-new-tokens 512 \
  --code-max-new-tokens 1024
```

## 常用参数
- `--ape-max-samples`: Ape210K 采样数，`<=0` 为全量
- `--apps-max-samples`: APPS 采样数，`<=0` 为全量
- `--apps-max-test-cases`: 每题最大测试用例数，`<=0` 为全量
- `--code-num-samples`: 用于 Pass@k 的候选数量（建议 10）
- `--code-timeout-s`: 单次代码执行超时


完全全量：
python /root/grpo/baseline/run_full_baseline.py \
    --datasets gsm8k,ape210k,math,cmath,mbpp,humaneval,apps \
    --ape-max-samples 0 \
    --apps-max-samples 0 \
    --apps-max-test-cases 0 \
    --code-num-samples 20 \
    --math-max-new-tokens 512 \
    --code-max-new-tokens 1024 \
    --output-root /root/autodl-tmp/baseline_results


如果是按照这个要求：
  - GSM8K：全量（~1K样本）
  - Ape210K：全量或采样（建议2-5K）
  - MATH：全量（~5K样本）
  - CMATH：全量
  - MBPP：全量（~500样本）
  - HumanEval：全量（164样本）
  - APPS：全量

python /root/grpo/baseline/run_full_baseline.py \
    --datasets gsm8k,ape210k,math,cmath,mbpp,humaneval,apps \
    --ape-max-samples 5000 \
    --apps-max-samples 0 \
    --apps-max-test-cases 0 \
    --code-num-samples 20 \
    --math-max-new-tokens 512 \
    --code-max-new-tokens 1024 \
    --output-root /root/autodl-tmp/baseline_results

如果你要 Ape210K 也全量，把 --ape-max-samples 5000 改成 --ape-max-samples 0。





  建议并行命令（同一张 GPU）：

  # 任务1：数学
  CUDA_VISIBLE_DEVICES=0 nohup python /root/grpo/baseline/run_full_baseline.py \
    --datasets gsm8k,ape210k,math,cmath \
    --ape-max-samples 2000 \
    --math-max-new-tokens 512 \
    --output-root /root/autodl-tmp/baseline_results_math \
    > /root/autodl-tmp/baseline_math.log 2>&1 &

  # 任务2：代码
  CUDA_VISIBLE_DEVICES=0 nohup python /root/grpo/baseline/run_full_baseline.py \
    --datasets mbpp,humaneval,apps \
    --apps-max-samples 500 \
    --apps-max-test-cases 5 \
    --code-num-samples 5 \
    --code-max-new-tokens 1024 \
    --output-root /root/autodl-tmp/baseline_results_code \
    > /root/autodl-tmp/baseline_code.log 2>&1 &

  查看进度：

  tail -f /root/autodl-tmp/baseline_math.log
  tail -f /root/autodl-tmp/baseline_code.log


  

