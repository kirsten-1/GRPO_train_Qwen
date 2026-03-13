# Stage1 GRPO Training

This folder contains a runnable Stage1 pipeline for simple math training.

## Files
- `prepare_stage1_data.py`: rebuild Stage1 data with configurable Ape210K:GSM8K ratio (default 3:1).
- `train_stage1_grpo.py`: Stage1 GRPO training with LoRA, quick/epoch/final eval, early stopping, and best checkpoint saving.
  - Default `--train-data` is `/root/autodl-tmp/training_data/stage1_simple_math_train_ratio3.json`.

## Dependencies
- `trl`
- `peft`
- `transformers`
- `datasets`

## 1) Prepare Stage1 data (recommended 3:1)
```bash
python /root/grpo/stage1/prepare_stage1_data.py \
  --processed-dir /root/autodl-tmp/processed_datasets \
  --ape-to-gsm-ratio 3 \
  --output /root/autodl-tmp/training_data/stage1_simple_math_train_ratio3.json
```

## 2) Dry-run checks before long training
```bash
python /root/grpo/stage1/train_stage1_grpo.py \
  --model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --train-data /root/autodl-tmp/training_data/stage1_simple_math_train_ratio3.json \
  --training-data-dir /root/autodl-tmp/training_data \
  --output-root /root/autodl-tmp/stage1_grpo_runs \
  --num-epochs 2 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --warmup-steps 100 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --group-size 4 \
  --kl-penalty 0.02 \
  --temperature 0.7 \
  --math-max-new-tokens 512 \
  --eval-every-steps 100 \
  --early-stop-gsm-gain 8 \
  --report-to tensorboard \
  --dry-run
```

## 3) Start training
```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/train_stage1_grpo.py \
  --model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct \
  --train-data /root/autodl-tmp/training_data/stage1_simple_math_train_ratio3.json \
  --training-data-dir /root/autodl-tmp/training_data \
  --output-root /root/autodl-tmp/stage1_grpo_runs \
  --num-epochs 2 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --warmup-steps 100 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj \
  --group-size 4 \
  --kl-penalty 0.02 \
  --temperature 0.7 \
  --math-max-new-tokens 512 \
  --code-max-new-tokens 512 \
  --use-vllm \
  --vllm-mode colocate \
  --vllm-gpu-memory-utilization 0.6 \
  --vllm-tensor-parallel-size 1 \
  --eval-every-steps 100 \
  --early-stop-gsm-gain 7 \
  --gradient-checkpointing \
  --report-to tensorboard
```

## Notes
- If memory is tight, use `--batch-size 4 --gradient-accumulation-steps 8`.
- For debugging or CPU-only fallback, disable vLLM with `--no-use-vllm`.
- Quick eval: gsm8k=50, ape210k=50, math=20, mbpp=10.
- Epoch eval: gsm8k=500, ape210k=1000, math=400, cmath=400, mbpp=100.
- Final eval: gsm8k=full, ape210k=full, math=200, cmath=100, mbpp=50.

## 4) vLLM Small-Sample Smoke Test (single-line commands)
Copy and run each command as one full line (do not manually insert line breaks).

```bash
python -c 'import json,random;src="/root/autodl-tmp/training_data/stage1_simple_math_train_ratio3.json";dst="/root/autodl-tmp/training_data/stage1_simple_math_train_ratio3_tiny128.json";data=json.load(open(src,"r",encoding="utf-8"));random.Random(42).shuffle(data);json.dump(data[:128],open(dst,"w",encoding="utf-8"),ensure_ascii=False,indent=2);print("wrote",dst,"samples=",len(data[:128]))'
```

```bash
CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/train_stage1_grpo.py --model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct --train-data /root/autodl-tmp/training_data/stage1_simple_math_train_ratio3_tiny128.json --training-data-dir /root/autodl-tmp/training_data --output-root /root/autodl-tmp/stage1_grpo_runs --run-name stage1_smoke_rewardcheck_vllm --num-epochs 1 --batch-size 2 --gradient-accumulation-steps 1 --group-size 2 --learning-rate 2e-5 --kl-penalty 0.02 --eval-every-steps 20 --math-max-new-tokens 256 --code-max-new-tokens 256 --report-to tensorboard --use-vllm --vllm-mode colocate --vllm-gpu-memory-utilization 0.6 --vllm-tensor-parallel-size 1
```

If GPU memory is tight, use this memory-optimized single-line variant:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python /root/grpo/stage1/train_stage1_grpo.py --model-path /root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct --train-data /root/autodl-tmp/training_data/stage1_simple_math_train_ratio3_tiny128.json --training-data-dir /root/autodl-tmp/training_data --output-root /root/autodl-tmp/stage1_grpo_runs --run-name stage1_smoke_rewardcheck_vllm_memopt --num-epochs 1 --batch-size 1 --gradient-accumulation-steps 2 --group-size 2 --learning-rate 2e-5 --kl-penalty 0.02 --eval-every-steps 20 --math-max-new-tokens 256 --code-max-new-tokens 256 --report-to tensorboard --use-vllm --vllm-mode colocate --vllm-gpu-memory-utilization 0.35 --vllm-tensor-parallel-size 1 --gradient-checkpointing
```

```bash
tensorboard --logdir /root/autodl-tmp/stage1_grpo_runs --port 6006 --bind_all
```

TensorBoard check:
- Open `http://127.0.0.1:6006` (or `http://<server-ip>:6006`).
- Go to `Scalars`, search `reward`, and inspect `reward/mean`.
- If `reward/mean` trends to `0.8-1.0+`, accuracy reward is likely active.
- If it stays around `0.2-0.4`, it often means only format/length rewards are active; verify whether `solution` is reaching TRL reward kwargs.

## 5) Migrate to Open-R1 Native Entry (vLLM Server Mode)
This path uses Open-R1's native GRPO entry (`open-r1/src/open_r1/grpo.py`) with separate vLLM server.

### 5.1 Convert Stage1 data to local Open-R1 dataset directory
```bash
python /root/grpo/stage1/prepare_stage1_openr1_dataset.py --input /root/autodl-tmp/training_data/stage1_simple_math_train_ratio3_tiny128.json --output-dir /root/autodl-tmp/training_data/stage1_openr1_dataset
```

For real training (not smoke), first rebalance to Ape:GSM=2:1 (GSM≈33%), then convert:
```bash
python /root/grpo/stage1/prepare_stage1_data.py --processed-dir /root/autodl-tmp/processed_datasets --ape-to-gsm-ratio 2.0 --output /root/autodl-tmp/training_data/stage1_simple_math_train_ratio2.json
```

```bash
python /root/grpo/stage1/prepare_stage1_openr1_dataset.py --input /root/autodl-tmp/training_data/stage1_simple_math_train_ratio2.json --output-dir /root/autodl-tmp/training_data/stage1_hf
```

### 5.2 One-command launch: start TRL vLLM server + run Open-R1 GRPO
```bash
bash /root/grpo/stage1/run_stage1_openr1_server.sh
```

Default behavior in `run_stage1_openr1_server.sh`:
- vLLM server on `SERVER_GPU=1`
- Training on `TRAIN_GPU=0`
- Config file: `/root/grpo/stage1/openr1_stage1_grpo_server.yaml`
- Server implementation: `launch_trl_vllm_server_compat.py` (internally uses TRL `vllm_serve` endpoints required by GRPO server mode)
- vLLM server args can be tuned by env vars:
  - `VLLM_GPU_MEMORY_UTILIZATION` (default `0.85`)
  - `VLLM_TENSOR_PARALLEL_SIZE` (default `1`)
  - `VLLM_MAX_MODEL_LEN` (default `8192`)

Optional override example:
```bash
SERVER_GPU=1 TRAIN_GPU=0 CFG=/root/grpo/stage1/openr1_stage1_grpo_server.yaml bash /root/grpo/stage1/run_stage1_openr1_server.sh
```

## 6) 1x5090 Pre-2x5090 Validation Checklist
Run these checks on single GPU before shutdown and switching to 2x5090.

### 6.1 Verify Open-R1 reward dependencies import correctly
```bash
PYTHONPATH=/root/grpo/open-r1/src python -c "from open_r1.rewards import accuracy_reward; print('rewards import ok')"
```

### 6.2 Rebuild tiny local Open-R1 dataset and inspect columns
```bash
python /root/grpo/stage1/prepare_stage1_openr1_dataset.py --input /root/autodl-tmp/training_data/stage1_simple_math_train_ratio3_tiny128.json --output-dir /root/autodl-tmp/training_data/stage1_openr1_dataset --max-samples 32 --shuffle
```

```bash
HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets python -c "import datasets;ds=datasets.load_dataset('/root/autodl-tmp/training_data/stage1_openr1_dataset');print(ds);print(ds['train'][0].keys());print(ds['train'][0]['problem'][:80]);print(ds['train'][0]['solution'][:80])"
```

### 6.3 Verify Open-R1 GRPO entry can parse args
```bash
cd /root/grpo/open-r1 && PYTHONPATH=/root/grpo/open-r1/src python src/open_r1/grpo.py --help | head -n 40
```

### 6.4 Run 3-step smoke training without vLLM (single GPU safe path)
```bash
cd /root/grpo/open-r1 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/root/grpo/open-r1/src HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file recipes/accelerate_configs/ddp.yaml --num_processes=1 src/open_r1/grpo.py --config /root/grpo/stage1/openr1_stage1_grpo_server.yaml --use_vllm false --attn_implementation sdpa --max_steps 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --num_generations 2 --max_prompt_length 512 --max_completion_length 64 --torch_empty_cache_steps 1
```

Notes:
- GRPO requires `num_generations >= 2`.
- Also ensure `per_device_train_batch_size * num_processes * gradient_accumulation_steps` is divisible by `num_generations`.

### 6.5 Optional: monitor GPU memory while smoke test is running
```bash
watch -n 1 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
```

Go/No-Go rule:
- If 6.1 to 6.4 all pass, environment and Open-R1 migration path are healthy enough to power off and switch to 2x5090.
- On 1x5090, do not treat vLLM server mode as final performance validation; do that after switching to 2x5090.

## 7) 2x5090 Final Preflight and Launch

### 7.1 Check dataset ratio (most important)
```bash
python - <<'PY'
import json
from collections import Counter
p='/root/autodl-tmp/training_data/stage1_hf/train.jsonl'
c=Counter();n=0
with open(p,'r',encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        obj=json.loads(line)
        c[obj.get('dataset','unknown')] += 1
        n += 1
print('total', n)
for k in sorted(c):
    print(f'{k}: {c[k]} ({c[k]/n*100:.2f}%)')
PY
```

Target:
- GSM8K >= 30%
- Ape210K <= 70%

### 7.2 Use the 2x5090 config
Use `/root/grpo/stage1/openr1_stage1_grpo_server_2x5090.yaml` for real 2x5090 runs.

### 7.3 Small validation run on 2x5090 (recommended first run after boot)
```bash
SERVER_GPU=1 TRAIN_GPU=0 CFG=/root/grpo/stage1/openr1_stage1_grpo_server_2x5090.yaml VLLM_GPU_MEMORY_UTILIZATION=0.75 VLLM_TENSOR_PARALLEL_SIZE=1 VLLM_MAX_MODEL_LEN=4096 bash /root/grpo/stage1/run_stage1_openr1_server.sh --max_steps 30 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --num_generations 2 --max_prompt_length 512 --max_completion_length 256 --save_strategy no --logging_steps 1
```

Pass criteria for the small validation run:
- vLLM server starts and health check passes.
- Training runs to `max_steps` without OOM/crash.
- TensorBoard has `reward`, `kl`, and completion length metrics.

### 7.4 Full run on 2x5090 (after validation passes)
```bash
SERVER_GPU=1 TRAIN_GPU=0 CFG=/root/grpo/stage1/openr1_stage1_grpo_server_2x5090.yaml VLLM_GPU_MEMORY_UTILIZATION=0.85 VLLM_TENSOR_PARALLEL_SIZE=1 VLLM_MAX_MODEL_LEN=8192 bash /root/grpo/stage1/run_stage1_openr1_server.sh
```
