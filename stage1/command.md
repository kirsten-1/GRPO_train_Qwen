# Stage1 GRPO Training Commands

## 方案 A: 1 epoch + batch_size=2 + grad_accum=2 (最优，4.8小时) ⭐推荐

**实测结果**：batch_size=8 反而更慢（14-15小时），因为 vLLM 生成是瓶颈。
保持原始 batch_size=2 是最优选择。

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

**配置说明:**
- Epochs: 1
- Batch size: 2
- Gradient accumulation: 2
- Effective batch size: 4
- 预计时间: 4.8 小时
- 速度: 3.08 秒/step

---

## 方案 B (不推荐): 1 epoch + batch_size=8 + grad_accum=2

**警告**：实测速度 10.5 秒/step，总时间 14-15 小时，比 batch_size=2 更慢！

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
  --per_device_train_batch_size 8 \
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

**警告**：实测速度 10.5 秒/step，总时间 14-15 小时，比 batch_size=2 更慢！

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
  --per_device_train_batch_size 8 \
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

---

## 方案 C: 2 epochs + batch_size=2 + grad_accum=2 (充分训练，9.6小时)

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
  --num_train_epochs 2 \
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

**配置说明:**
- Epochs: 2
- Batch size: 2
- Gradient accumulation: 2
- Effective batch size: 4
- 预计时间: 9.6 小时

---

## 性能分析总结

**实测结果**：
- batch_size=2: 3.08 秒/step ✅ 最快
- batch_size=8: 10.5 秒/step ❌ 慢 3.4 倍

**原因**：vLLM 生成是瓶颈，batch_size 越大，生成时间越长，无法线性扩展。

**推荐**：使用方案 A（1 epoch, batch_size=2），4.8 小时完成。

---

## 当前运行配置 (原始，已废弃)

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
  --num_train_epochs 2 \
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

**配置说明:**
- Epochs: 2
- Batch size: 2
- Gradient accumulation: 2
- Effective batch size: 4
- 预计时间: 38.4 小时

---

## 使用说明

### 首次运行或更改 batch_size 时

**重要**：如果更改了 batch_size，必须清理旧的 checkpoint，否则会出现张量维度不匹配错误。

```bash
# 方法 1: 备份旧目录（推荐）
cd /root/autodl-tmp/stage1_grpo_runs
mv openr1_stage1_server_2x5090 openr1_stage1_server_2x5090_backup_$(date +%Y%m%d_%H%M%S)
mkdir -p openr1_stage1_server_2x5090

# 方法 2: 直接删除旧 checkpoint
rm -rf /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090/checkpoint-*
```

### 运行步骤

1. **停止当前训练** (如果正在运行):
   ```bash
   pkill -f "open_r1/grpo.py"
   pkill -f "launch_trl_vllm_server"
   ```

2. **清理旧 checkpoint**（如果更改了 batch_size）:
   ```bash
   cd /root/autodl-tmp/stage1_grpo_runs
   mv openr1_stage1_server_2x5090 openr1_stage1_server_2x5090_old
   mkdir -p openr1_stage1_server_2x5090
   ```

3. **选择并运行方案**:
   - 推荐方案 A (最快，4.8小时)
   - 或方案 C (更充分训练，9.6小时)

4. **监控训练**:
   ```bash
   tail -f /tmp/stage1_train_sync.log
   ```

5. **检查 GPU 使用**:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

## LoRA 合并（用于评估）

```bash
python /root/grpo/stage1/merge_lora.py \
  --base-model /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --adapter-path /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090 \
  --output-dir /root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
  --dtype bfloat16 \
  --device-map auto
```

---

## 评估（小样本 / baseline 对齐 / full）

### 1) 小样本 smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
CODE_BENCHMARKS=none \
bash /root/grpo/stage1/run_stage1_eval_smoke.sh
```

### 2) baseline 对齐（推荐汇报口径）

baseline 对齐规模：
- GSM8K: 1319
- Ape210K: 2000
- MATH: 1187
- CMATH: 1098

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
CODE_BENCHMARKS=none \
bash /root/grpo/stage1/run_stage1_eval_baseline.sh
```

### 3) full 规模

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
CODE_BENCHMARKS=mbpp \
bash /root/grpo/stage1/run_stage1_eval_full.sh
```

### 4) 可选代码评估（防灾难遗忘）

```bash
# MBPP only
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
CODE_BENCHMARKS=mbpp \
bash /root/grpo/stage1/run_stage1_eval_baseline.sh
```

```bash
# MBPP + HumanEval proxy
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
CODE_BENCHMARKS=mbpp_humaneval \
bash /root/grpo/stage1/run_stage1_eval_baseline.sh
```

```bash
# MBPP + HumanEval proxy + APPS proxy
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090_merged \
CODE_BENCHMARKS=all \
bash /root/grpo/stage1/run_stage1_eval_baseline.sh
```

评估结果统一输出到 `/root/autodl-tmp/stage1_eval_runs/*/eval_results.json`。
