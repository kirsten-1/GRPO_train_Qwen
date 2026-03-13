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

```bash
tensorboard --logdir /root/autodl-tmp/stage1_grpo_runs --port 6006 --bind_all
```

TensorBoard check:
- Open `http://127.0.0.1:6006` (or `http://<server-ip>:6006`).
- Go to `Scalars`, search `reward`, and inspect `reward/mean`.
- If `reward/mean` trends to `0.8-1.0+`, accuracy reward is likely active.
- If it stays around `0.2-0.4`, it often means only format/length rewards are active; verify whether `solution` is reaching TRL reward kwargs.
