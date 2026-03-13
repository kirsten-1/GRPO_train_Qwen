# Stage1 GRPO Training Commands

## 方案 A: 1 epoch + batch_size=8 + grad_accum=2 (推荐，4.8小时)

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

**配置说明:**
- Epochs: 1
- Batch size: 8
- Gradient accumulation: 2
- Effective batch size: 16
- 预计时间: 4.8 小时
- 加速比: 8x

---

## 方案 C: 2 epochs + batch_size=8 + grad_accum=2 (平衡，9.6小时)

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

**配置说明:**
- Epochs: 2
- Batch size: 8
- Gradient accumulation: 2
- Effective batch size: 16
- 预计时间: 9.6 小时
- 加速比: 4x

---

## 当前运行配置 (原始，38.4小时)

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

1. **停止当前训练** (如果正在运行):
   ```bash
   pkill -f "open_r1/grpo.py"
   pkill -f "launch_trl_vllm_server"
   ```

2. **选择并运行方案**:
   - 推荐方案 A (最快)
   - 或方案 C (更充分训练)

3. **监控训练**:
   ```bash
   tail -f /tmp/stage1_train_sync.log
   ```

4. **检查 GPU 使用**:
   ```bash
   watch -n 1 nvidia-smi
   ```
