# Stage2 Commands (Hard Math)

## 0) Prepare Stage2 HF Dataset

```bash
python /root/grpo/stage2/prepare_stage2_openr1_dataset.py \
  --input /root/autodl-tmp/training_data/stage2_hard_math_train.json \
  --output-dir /root/autodl-tmp/training_data/stage2_hf
```

默认会过滤为 `math` only。若需要混入 Stage1 数据，可额外加 `--mix-stage1-ratio 0.10`。

---

## 1) Stage2 Train - Epoch 1 (separate command)

先找 Stage1 最新 checkpoint：

```bash
CKPT="$(find /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
echo "$CKPT"
```

再计算推荐 `max_steps`（从该 checkpoint 增量跑 1 个 Stage2 epoch）：

```bash
python /root/grpo/stage2/calc_stage2_max_steps.py \
  --dataset-jsonl /root/autodl-tmp/training_data/stage2_hf/train.jsonl \
  --resume-checkpoint "$CKPT" \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --mode observed \
  --epochs-to-run 1
```

按输出里的 `recommended_max_steps` 启动训练（下面用环境变量注入）：

```bash
TRL_VLLM_GROUP_PORT=55693 \
TRL_VLLM_SKIP_BARRIER=1 \
TRL_VLLM_SYNC_TRACE=0 \
TRL_VLLM_SYNC_BARRIER_INTERVAL=0 \
TRL_VLLM_SYNC_TRAINABLE_ONLY=1 \
TRL_VLLM_SYNC_PEF_TARGET_ONLY=1 \
TRL_VLLM_SYNC_BEFORE_UNMERGE=1 \
TRL_VLLM_SKIP_UNMERGE=0 \
TRL_VLLM_SKIP_RESET_PREFIX_CACHE=1 \
TRL_VLLM_UPDATE_STRICT=0 \
TRL_VLLM_REQUEST_TIMEOUT=240 \
TRL_VLLM_GENERATE_RECV_TIMEOUT=120 \
VLLM_ENFORCE_EAGER=1 \
ACCELERATE_LOG_LEVEL=warning \
TRANSFORMERS_VERBOSITY=warning \
SERVER_GPU=1 \
TRAIN_GPU=0 \
CFG=/root/grpo/stage2/openr1_stage2_grpo_server_2x5090.yaml \
VLLM_GPU_MEMORY_UTILIZATION=0.55 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_MAX_MODEL_LEN=2048 \
MAX_STEPS=26088 \
bash /root/grpo/stage2/run_stage2_openr1_server.sh \
  --resume_from_checkpoint "$CKPT" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_generations 4 \
  --generation_batch_size 8 \
  --max_prompt_length 1024 \
  --max_completion_length 512 \
  --beta 0.10 \
  --learning_rate 5e-6 \
  --max_grad_norm 0.5 \
  --temperature 0.7 \
  --warmup_steps 50 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 8 \
  --logging_steps 10 \
  --max_steps "${MAX_STEPS}" \
  2>&1 | tee /tmp/stage2_train_epoch1.log
```

也可以直接一键脚本（自动计算 `max_steps`）：

```bash
bash /root/grpo/stage2/run_stage2_epoch1.sh
```

脚本默认 `STRICT_MATH_ONLY=1`，检测到训练集含 `cmath/gsm8k/ape210k` 会直接退出，防止误训。
训练使用 `num_generations=4` 时，`temperature` 必须 `>0`（vLLM 限制，`temperature=0` 仅适用于 `num_generations=1`）。

实时监控（loss/reward/kl/lr）：

```bash
tail -f /tmp/stage2_train_epoch1.log | rg --line-buffered "'loss'|'reward'|'kl'|'learning_rate'"
```

---

## 2) Stage2 Eval (separate command)

### 2.1 核心 + 保持能力（推荐）

```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite full \
  --code-num-samples 5 \
  --timeout-s 6 \
  --mbpp-samples 257 \
  --humaneval-samples 164 \
  --progress-every 10 \
  --run-name stage2_epoch1_full
```

说明：数学任务评估固定 `do_sample=False`（greedy）；`temperature` 只影响代码 `pass@k` 采样。

等价一键脚本：

```bash
bash /root/grpo/stage2/run_stage2_eval_full.sh
```

### 2.2 只看困难数学（MATH+CMATH）

```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite core \
  --progress-every 10 \
  --run-name stage2_epoch1_core
```

### 2.3 每500步做一次 GSM8K quick eval（手动触发）

```bash
GSM8K_SAMPLES=500 \
MBPP_SAMPLES=257 \
HUMANEVAL_SAMPLES=164 \
CODE_NUM_SAMPLES=3 \
TIMEOUT_S=6 \
bash /root/grpo/stage2/run_stage2_eval_quick_gsm.sh
```

### 2.4 自动守护（检测新 checkpoint 就跑 GSM8K quick eval）

```bash
GSM8K_REF_ACC=57.54 \
GSM8K_WARN_DROP=3 \
GSM8K_STOP_DROP=5 \
CODE_GUARD_ENABLED=1 \
MBPP_SAMPLES=257 \
HUMANEVAL_SAMPLES=164 \
CODE_NUM_SAMPLES=3 \
CODE_TIMEOUT_S=6 \
STOP_ON_DROP=1 \
EVAL_GPU=1 \
TRAIN_PID_FILE=/tmp/stage2_train_epoch1.pid \
bash /root/grpo/stage2/watch_stage2_gsm_guard.sh
```

如果不希望触发阈值后自动停训，把 `STOP_ON_DROP=0`。

### 2.5 与 Stage1 数学评估严格对齐（推荐用于横向对比）

使用和 `stage1/eval_stage1_model.py --suite baseline` 一致的样本规模：
- GSM8K=1319
- Ape210K=2000
- MATH=1187
- CMATH=1098

```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite stage1_math_aligned \
  --code-num-samples 0 \
  --progress-every 10 \
  --run-name stage2_math_aligned_to_stage1
```

### 2.6 与代码 Baseline 严格对齐（MBPP/HumanEval）

使用 `code_math_baseline_data_summary.md` 同配置：
- MBPP=257
- HumanEval=164
- code_num_samples=5
- timeout=6

```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite code_baseline_aligned \
  --code-num-samples 5 \
  --timeout-s 6 \
  --progress-every 10 \
  --run-name stage2_code_aligned_to_baseline
```

---

## 3) 训练后汇总与决策

汇总训练动态（KL/Loss/Reward/LR）：

```bash
python /root/grpo/stage2/summarize_stage2_training.py \
  --trainer-state /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090/trainer_state.json \
  --output /root/autodl-tmp/stage2_eval_runs/stage2_train_summary.json
```

基于 Stage1 vs Stage2 评估做决策：

```bash
python /root/grpo/stage2/decide_stage2_next.py \
  --stage1-eval /path/to/stage1_eval_results.json \
  --stage2-eval /path/to/stage2_eval_results.json \
  --gsm8k-max-drop 5 \
  --ape210k-max-drop 5 \
  --output /root/autodl-tmp/stage2_eval_runs/stage2_decision.json
```

---

## 4) Epoch 2/3 继续训练

思路：从 Stage2 最新 checkpoint 继续，重新计算 `recommended_max_steps` 再启动训练命令（只改 `--resume_from_checkpoint` 和 `--max_steps`）。

如果出现明显遗忘：
- `GSM8K` 下降 5-10%：优先把 `learning_rate` 降到 `3e-6`
- `GSM8K` 下降 >10%：停止当前配置，考虑混入 10-20% 简单数学数据再训
