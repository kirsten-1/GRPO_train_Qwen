#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/grpo}"
CFG="${CFG:-$ROOT/stage2/openr1_stage2_grpo_server_2x5090.yaml}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-/root/autodl-tmp/stage1_grpo_runs/openr1_stage1_server_2x5090}"
RESUME_CKPT="${RESUME_CKPT:-}"
DATASET_JSONL="${DATASET_JSONL:-/root/autodl-tmp/training_data/stage2_hf/train.jsonl}"
EPOCHS_TO_RUN="${EPOCHS_TO_RUN:-1}"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-/tmp/stage2_train_epoch1.pid}"
STRICT_MATH_ONLY="${STRICT_MATH_ONLY:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.7}"

echo "$$" > "$TRAIN_PID_FILE"
trap 'rm -f "$TRAIN_PID_FILE"' EXIT

if [[ -z "${RESUME_CKPT}" ]]; then
  RESUME_CKPT="$(find "$STAGE1_RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi
if [[ -z "${RESUME_CKPT}" || ! -d "${RESUME_CKPT}" ]]; then
  echo "Invalid RESUME_CKPT: ${RESUME_CKPT}" >&2
  exit 1
fi

MAX_STEPS="$(python "$ROOT/stage2/calc_stage2_max_steps.py" \
  --dataset-jsonl "$DATASET_JSONL" \
  --resume-checkpoint "$RESUME_CKPT" \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --mode observed \
  --epochs-to-run "$EPOCHS_TO_RUN" | awk -F= '/recommended_max_steps/{print $2}')"

if [[ -z "${MAX_STEPS}" ]]; then
  echo "Failed to compute MAX_STEPS" >&2
  exit 1
fi

if [[ "${STRICT_MATH_ONLY}" == "1" ]]; then
  if ! python - "$DATASET_JSONL" <<'PY'
import json,sys
from collections import Counter
p=sys.argv[1]
counts=Counter()
with open(p,"r",encoding="utf-8") as f:
  for line in f:
    if not line.strip():
      continue
    obj=json.loads(line)
    counts[str(obj.get("dataset","unknown")).strip().lower()] += 1
bad={k:v for k,v in counts.items() if k!="math"}
if bad:
  print(f"[stage2] STRICT_MATH_ONLY=1 but found non-math samples: {bad}", file=sys.stderr)
  sys.exit(2)
print(f"[stage2] dataset check passed (math-only): {dict(counts)}")
PY
  then
    echo "[stage2] Rebuild dataset with:" >&2
    echo "python /root/grpo/stage2/prepare_stage2_openr1_dataset.py --input /root/autodl-tmp/training_data/stage2_hard_math_train.json --output-dir /root/autodl-tmp/training_data/stage2_hf --only-math" >&2
    exit 1
  fi
fi

if ! python - "$NUM_GENERATIONS" "$TRAIN_TEMPERATURE" <<'PY'
import sys
n=int(float(sys.argv[1]))
t=float(sys.argv[2])
if n > 1 and t <= 0.0:
    print(
        f"[stage2] invalid config: num_generations={n} with temperature={t}. "
        "vLLM greedy mode requires n==1.",
        file=sys.stderr,
    )
    sys.exit(2)
PY
then
  echo "[stage2] Use TRAIN_TEMPERATURE>0 (e.g. 0.7) or set NUM_GENERATIONS=1." >&2
  exit 1
fi

echo "Using RESUME_CKPT=${RESUME_CKPT}"
echo "Using MAX_STEPS=${MAX_STEPS}"
echo "Wrote TRAIN_PID_FILE=${TRAIN_PID_FILE} (pid=$$)"
echo "Using NUM_GENERATIONS=${NUM_GENERATIONS} TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE}"

TRL_VLLM_GROUP_PORT="${TRL_VLLM_GROUP_PORT:-55693}" \
TRL_VLLM_SKIP_BARRIER="${TRL_VLLM_SKIP_BARRIER:-1}" \
TRL_VLLM_SYNC_TRACE="${TRL_VLLM_SYNC_TRACE:-0}" \
TRL_VLLM_SYNC_BARRIER_INTERVAL="${TRL_VLLM_SYNC_BARRIER_INTERVAL:-0}" \
TRL_VLLM_SYNC_TRAINABLE_ONLY="${TRL_VLLM_SYNC_TRAINABLE_ONLY:-1}" \
TRL_VLLM_SYNC_PEF_TARGET_ONLY="${TRL_VLLM_SYNC_PEF_TARGET_ONLY:-1}" \
TRL_VLLM_SYNC_BEFORE_UNMERGE="${TRL_VLLM_SYNC_BEFORE_UNMERGE:-1}" \
TRL_VLLM_SKIP_UNMERGE="${TRL_VLLM_SKIP_UNMERGE:-0}" \
TRL_VLLM_SKIP_RESET_PREFIX_CACHE="${TRL_VLLM_SKIP_RESET_PREFIX_CACHE:-1}" \
TRL_VLLM_UPDATE_STRICT="${TRL_VLLM_UPDATE_STRICT:-0}" \
TRL_VLLM_REQUEST_TIMEOUT="${TRL_VLLM_REQUEST_TIMEOUT:-240}" \
TRL_VLLM_GENERATE_RECV_TIMEOUT="${TRL_VLLM_GENERATE_RECV_TIMEOUT:-120}" \
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}" \
ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-warning}" \
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}" \
SERVER_GPU="${SERVER_GPU:-1}" \
TRAIN_GPU="${TRAIN_GPU:-0}" \
CFG="$CFG" \
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}" \
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}" \
bash "$ROOT/stage2/run_stage2_openr1_server.sh" \
  --resume_from_checkpoint "$RESUME_CKPT" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_generations "$NUM_GENERATIONS" \
  --generation_batch_size 8 \
  --max_prompt_length 1024 \
  --max_completion_length 512 \
  --beta 0.10 \
  --learning_rate 5e-6 \
  --max_grad_norm 0.5 \
  --temperature "$TRAIN_TEMPERATURE" \
  --warmup_steps 50 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 8 \
  --logging_steps 10 \
  --max_steps "$MAX_STEPS" \
  2>&1 | tee /tmp/stage2_train_epoch1.log
