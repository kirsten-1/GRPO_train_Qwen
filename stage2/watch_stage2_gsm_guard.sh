#!/usr/bin/env bash
set -euo pipefail

# Poll Stage2 checkpoints and run quick eval per checkpoint.
# Monitors GSM8K sentinel and optional MBPP/HumanEval retention.

MODEL_BASE_PATH="${MODEL_BASE_PATH:-/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/autodl-tmp/stage2_grpo_runs/openr1_stage2_server_2x5090}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/stage2_eval_runs}"
EVAL_GPU="${EVAL_GPU:-1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-500}"
GSM8K_REF_ACC="${GSM8K_REF_ACC:-57.54}"      # percentage
GSM8K_WARN_DROP="${GSM8K_WARN_DROP:-3}"      # percentage points
GSM8K_STOP_DROP="${GSM8K_STOP_DROP:-5}"      # percentage points
CODE_GUARD_ENABLED="${CODE_GUARD_ENABLED:-1}" # 1 -> track MBPP/HumanEval in quick eval
MBPP_SAMPLES="${MBPP_SAMPLES:-257}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-164}"
CODE_NUM_SAMPLES="${CODE_NUM_SAMPLES:-3}"
CODE_TIMEOUT_S="${CODE_TIMEOUT_S:-6}"
CODE_TEMPERATURE="${CODE_TEMPERATURE:-0.7}"
CODE_TOP_P="${CODE_TOP_P:-0.95}"
MBPP_WARN_PASS1="${MBPP_WARN_PASS1:-0.50}"         # absolute pass@1
MBPP_STOP_PASS1="${MBPP_STOP_PASS1:-0.45}"         # absolute pass@1
HUMANEVAL_WARN_PASS1="${HUMANEVAL_WARN_PASS1:-0.35}" # absolute pass@1
HUMANEVAL_STOP_PASS1="${HUMANEVAL_STOP_PASS1:-0.30}" # absolute pass@1
STOP_ON_DROP="${STOP_ON_DROP:-1}"            # 1 -> kill training process
TRAIN_PID_FILE="${TRAIN_PID_FILE:-}"         # optional pid file for precise stop

STATE_FILE="${STATE_FILE:-/tmp/stage2_gsm_guard_seen.txt}"
touch "$STATE_FILE"

echo "[guard] checkpoint_root=$CHECKPOINT_ROOT eval_gpu=$EVAL_GPU poll=${POLL_SECONDS}s"
echo "[guard] gsm_ref=${GSM8K_REF_ACC}% warn_drop=${GSM8K_WARN_DROP} stop_drop=${GSM8K_STOP_DROP} stop_on_drop=${STOP_ON_DROP}"
echo "[guard] code_guard=$CODE_GUARD_ENABLED mbpp_samples=$MBPP_SAMPLES humaneval_samples=$HUMANEVAL_SAMPLES code_num_samples=$CODE_NUM_SAMPLES"

while true; do
  mapfile -t CKPTS < <(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  for ckpt in "${CKPTS[@]}"; do
    if grep -qxF "$ckpt" "$STATE_FILE"; then
      continue
    fi
    echo "$ckpt" >> "$STATE_FILE"
    step="${ckpt##*-}"
    echo "[guard] evaluating $ckpt"

    if [[ "$CODE_GUARD_ENABLED" == "1" ]]; then
      mbpp_samples="$MBPP_SAMPLES"
      humaneval_samples="$HUMANEVAL_SAMPLES"
    else
      mbpp_samples="0"
      humaneval_samples="0"
    fi

    RUN_NAME="stage2_ckpt_${step}_gsm_guard"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python /root/grpo/stage2/eval_stage2_model.py \
      --model-path "$ckpt" \
      --base-model-path "$MODEL_BASE_PATH" \
      --output-root "$OUTPUT_ROOT" \
      --suite retain \
      --gsm8k-samples "$GSM8K_SAMPLES" \
      --ape210k-samples 0 \
      --math-samples 0 \
      --cmath-samples 0 \
      --mbpp-samples "$mbpp_samples" \
      --humaneval-samples "$humaneval_samples" \
      --code-num-samples "$CODE_NUM_SAMPLES" \
      --timeout-s "$CODE_TIMEOUT_S" \
      --temperature "$CODE_TEMPERATURE" \
      --top-p "$CODE_TOP_P" \
      --progress-every 10 \
      --run-name "$RUN_NAME"

    eval_json="$(find "$OUTPUT_ROOT" -maxdepth 2 -type f -path "*/${RUN_NAME}_retain_*/eval_results.json" | sort | tail -n 1)"
    if [[ -z "${eval_json}" ]]; then
      echo "[guard] WARN: no eval_results.json found after checkpoint $step"
      continue
    fi
    read -r gsm_acc mbpp_pass1 humaneval_pass1 < <(python - <<PY
import json
d=json.load(open("$eval_json"))
m=d.get("metrics",{})
gsm=float(m.get("gsm8k_accuracy",0.0))*100.0
mbpp=float(m.get("mbpp_pass@1", m.get("mbpp_pass1", 0.0)))*100.0
hum=float(m.get("humaneval_pass@1", m.get("humaneval_pass1", 0.0)))*100.0
print(f"{gsm} {mbpp} {hum}")
PY
)
    drop="$(python -c "print(float('${GSM8K_REF_ACC}')-float('${gsm_acc}'))")"
    echo "[guard] checkpoint=$step gsm8k_acc=${gsm_acc}% drop=${drop}pp mbpp_pass1=${mbpp_pass1}% humaneval_pass1=${humaneval_pass1}%"

    python - <<PY
drop=float("${drop}")
warn=float("${GSM8K_WARN_DROP}")
stop=float("${GSM8K_STOP_DROP}")
code_guard=int("${CODE_GUARD_ENABLED}")
mbpp=float("${mbpp_pass1}")/100.0
hum=float("${humaneval_pass1}")/100.0
mbpp_warn=float("${MBPP_WARN_PASS1}")
mbpp_stop=float("${MBPP_STOP_PASS1}")
hum_warn=float("${HUMANEVAL_WARN_PASS1}")
hum_stop=float("${HUMANEVAL_STOP_PASS1}")
if drop > stop:
    print("[guard] ALERT: GSM8K drop > stop threshold")
elif drop > warn:
    print("[guard] WARN: GSM8K drop > warn threshold")
if code_guard:
    if mbpp < mbpp_stop:
        print("[guard] ALERT: MBPP pass@1 < stop threshold")
    elif mbpp < mbpp_warn:
        print("[guard] WARN: MBPP pass@1 < warn threshold")
    if hum < hum_stop:
        print("[guard] ALERT: HumanEval pass@1 < stop threshold")
    elif hum < hum_warn:
        print("[guard] WARN: HumanEval pass@1 < warn threshold")
PY

    stop_hit="$(python - <<PY
drop=float("${drop}")
gsm_stop=float("${GSM8K_STOP_DROP}")
code_guard=int("${CODE_GUARD_ENABLED}")
mbpp=float("${mbpp_pass1}")/100.0
hum=float("${humaneval_pass1}")/100.0
mbpp_stop=float("${MBPP_STOP_PASS1}")
hum_stop=float("${HUMANEVAL_STOP_PASS1}")
hit = (drop > gsm_stop)
if code_guard:
    hit = hit or (mbpp < mbpp_stop) or (hum < hum_stop)
print(1 if hit else 0)
PY
)"
    if [[ "$STOP_ON_DROP" == "1" && "$stop_hit" == "1" ]]; then
      echo "[guard] STOP_ON_DROP=1 and stop threshold hit."
      if [[ -n "${TRAIN_PID_FILE}" && -f "${TRAIN_PID_FILE}" ]]; then
        TRAIN_PID="$(cat "$TRAIN_PID_FILE" 2>/dev/null || true)"
        if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" >/dev/null 2>&1; then
          echo "[guard] killing process-group -${TRAIN_PID} from ${TRAIN_PID_FILE}"
          kill -- "-${TRAIN_PID}" || kill "${TRAIN_PID}" || true
        else
          echo "[guard] pid file found but process not alive: ${TRAIN_PID_FILE}"
        fi
      else
        echo "[guard] TRAIN_PID_FILE not set; skip kill to avoid broad termination."
      fi
      exit 2
    fi
  done
  sleep "$POLL_SECONDS"
done
