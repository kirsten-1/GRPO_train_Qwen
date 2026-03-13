# GRPO 运行命令总览（Baseline / Stage1 / Stage2）

## 0) 说明
- 路径：`/root/grpo/readme.md`
- 评估脚本已修复为与训练一致的数学判定逻辑（`math_verify` 路径）。
- 评估会保存全部样本记录：
  - `eval_results.json`（含 `sample_generations`）
  - `sample_generations.jsonl`（全量逐条）

---

## 1) Baseline（基座模型）评估

### 1.1 单命令（baseline 套件）
```bash
CUDA_VISIBLE_DEVICES=0 CODE_BENCHMARKS=none bash /root/grpo/stage1/run_stage1_eval_baseline.sh
```

### 1.2 四个数据集并行（4 终端）

```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --output-root /root/autodl-tmp/stage1_eval_runs \
  --run-name baseline_reval_gsm8k \
  --suite baseline \
  --code-benchmarks none \
  --gsm8k-samples 1319 \
  --ape210k-samples 0 \
  --math-samples 0 \
  --cmath-samples 0 \
  --mbpp-samples 0 \
  --humaneval-samples 0 \
  --apps-samples 0 \
  --progress-bar \
  --progress-every 10 \
  2>&1 | tee /tmp/baseline_reval_gsm8k.log
```

```bash
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --output-root /root/autodl-tmp/stage1_eval_runs \
  --run-name baseline_reval_ape210k \
  --suite baseline \
  --code-benchmarks none \
  --gsm8k-samples 0 \
  --ape210k-samples 2000 \
  --math-samples 0 \
  --cmath-samples 0 \
  --mbpp-samples 0 \
  --humaneval-samples 0 \
  --apps-samples 0 \
  --progress-bar \
  --progress-every 10 \
  2>&1 | tee /tmp/baseline_reval_ape210k.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --output-root /root/autodl-tmp/stage1_eval_runs \
  --run-name baseline_reval_math \
  --suite baseline \
  --code-benchmarks none \
  --gsm8k-samples 0 \
  --ape210k-samples 0 \
  --math-samples 1187 \
  --cmath-samples 0 \
  --mbpp-samples 0 \
  --humaneval-samples 0 \
  --apps-samples 0 \
  --progress-bar \
  --progress-every 10 \
  2>&1 | tee /tmp/baseline_reval_math.log
```

```bash
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --output-root /root/autodl-tmp/stage1_eval_runs \
  --run-name baseline_reval_cmath \
  --suite baseline \
  --code-benchmarks none \
  --gsm8k-samples 0 \
  --ape210k-samples 0 \
  --math-samples 0 \
  --cmath-samples 1098 \
  --mbpp-samples 0 \
  --humaneval-samples 0 \
  --apps-samples 0 \
  --progress-bar \
  --progress-every 10 \
  2>&1 | tee /tmp/baseline_reval_cmath.log
```

---

## 2) Stage1

### 2.1 训练（1 epoch，推荐）
```bash
TRL_VLLM_GROUP_PORT=55692 \
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
CFG=/root/grpo/stage1/openr1_stage1_grpo_server_2x5090.yaml \
VLLM_GPU_MEMORY_UTILIZATION=0.55 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_MAX_MODEL_LEN=2048 \
bash /root/grpo/stage1/run_stage1_openr1_server.sh \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_generations 4 \
  --generation_batch_size 8 \
  --max_prompt_length 1024 \
  --max_completion_length 512 \
  --beta 0.08 \
  --learning_rate 1e-5 \
  --max_grad_norm 0.5 \
  --temperature 0.7 \
  --warmup_steps 100 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --logging_steps 10 \
  2>&1 | tee /tmp/stage1_train_sync.log
```

### 2.2 Merge（评估 merged 模型）
```bash
python /root/grpo/stage1/merge_lora.py \
  --base-model /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --adapter-path /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090 \
  --output-dir /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
  --dtype bfloat16 \
  --device-map auto
```

### 2.3 评估（Stage1 训练后模型，4 终端并行）

```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
  --output-root /root/autodl-tmp/stage1_eval_runs \
  --run-name stage1_reval_gsm8k \
  --suite baseline \
  --code-benchmarks none \
  --gsm8k-samples 1319 \
  --ape210k-samples 0 \
  --math-samples 0 \
  --cmath-samples 0 \
  --mbpp-samples 0 \
  --humaneval-samples 0 \
  --apps-samples 0 \
  --progress-bar \
  --progress-every 10 \
  2>&1 | tee /tmp/stage1_reval_gsm8k.log
```

```bash
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
  --output-root /root/autodl-tmp/stage1_eval_runs \
  --run-name stage1_reval_ape210k \
  --suite baseline \
  --code-benchmarks none \
  --gsm8k-samples 0 \
  --ape210k-samples 2000 \
  --math-samples 0 \
  --cmath-samples 0 \
  --mbpp-samples 0 \
  --humaneval-samples 0 \
  --apps-samples 0 \
  --progress-bar \
  --progress-every 10 \
  2>&1 | tee /tmp/stage1_reval_ape210k.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
  --output-root /root/autodl-tmp/stage1_eval_runs \
  --run-name stage1_reval_math \
  --suite baseline \
  --code-benchmarks none \
  --gsm8k-samples 0 \
  --ape210k-samples 0 \
  --math-samples 1187 \
  --cmath-samples 0 \
  --mbpp-samples 0 \
  --humaneval-samples 0 \
  --apps-samples 0 \
  --progress-bar \
  --progress-every 10 \
  2>&1 | tee /tmp/stage1_reval_math.log
```

```bash
CUDA_VISIBLE_DEVICES=1 python /root/grpo/stage1/eval_stage1_model.py \
  --model-path /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
  --output-root /root/autodl-tmp/stage1_eval_runs \
  --run-name stage1_reval_cmath \
  --suite baseline \
  --code-benchmarks none \
  --gsm8k-samples 0 \
  --ape210k-samples 0 \
  --math-samples 0 \
  --cmath-samples 1098 \
  --mbpp-samples 0 \
  --humaneval-samples 0 \
  --apps-samples 0 \
  --progress-bar \
  --progress-every 10 \
  2>&1 | tee /tmp/stage1_reval_cmath.log
```

---

## 3) Stage2

### 3.1 数据准备
```bash
python /root/grpo/stage2/prepare_stage2_openr1_dataset.py \
  --input /root/autodl-tmp/training_data/stage2_hard_math_train.json \
  --output-dir /root/autodl-tmp/training_data/stage2_hf
```

### 3.2 训练（Epoch1，一键）
```bash
bash /root/grpo/stage2/run_stage2_epoch1.sh
```

### 3.3 评估（full）
```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite full \
  --mbpp-samples 26 \
  --progress-every 10 \
  --run-name stage2_epoch1_full
```

### 3.4 评估（只看困难数学）
```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage2/eval_stage2_model.py \
  --model-path /root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090 \
  --base-model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --suite core \
  --progress-every 10 \
  --run-name stage2_epoch1_core
```

### 3.5 快速遗忘检测（GSM8K）
```bash
bash /root/grpo/stage2/run_stage2_eval_quick_gsm.sh
```

