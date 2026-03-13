#!/usr/bin/env bash
set -euo pipefail

# One-command runner:
# 1) start vLLM server on one GPU
# 2) launch Open-R1 native GRPO entry on another GPU in server mode

ROOT="${ROOT:-/root/grpo}"
OPENR1_ROOT="${OPENR1_ROOT:-$ROOT/open-r1}"
CFG="${CFG:-$ROOT/stage1/openr1_stage1_grpo_server.yaml}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen/Qwen2___5-3B-Instruct}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVER_GPU="${SERVER_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
TRL_VLLM_GROUP_PORT="${TRL_VLLM_GROUP_PORT:-$((50000 + RANDOM % 10000))}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/hf_cache/datasets}"
# Keep stage1 path in PYTHONPATH so sitecustomize.py is auto-imported in all spawned Python processes.
export PYTHONPATH="${ROOT}/stage1:${OPENR1_ROOT}/src:${PYTHONPATH:-}"
# Conservative NCCL defaults for cross-process weight sync stability (can be overridden by caller env).
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TRL_VLLM_SKIP_BARRIER="${TRL_VLLM_SKIP_BARRIER:-1}"
export TRL_VLLM_SYNC_TRACE="${TRL_VLLM_SYNC_TRACE:-0}"
export TRL_VLLM_SYNC_BARRIER_INTERVAL="${TRL_VLLM_SYNC_BARRIER_INTERVAL:-1}"
export TRL_VLLM_SYNC_TRAINABLE_ONLY="${TRL_VLLM_SYNC_TRAINABLE_ONLY:-1}"
export TRL_VLLM_GROUP_PORT
ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-warning}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"
mkdir -p "$HF_DATASETS_CACHE"

python - <<'PY'
import importlib.util
import sys

required = ["latex2sympy2_extended", "math_verify"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python packages:", ", ".join(missing))
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)
PY

echo "[1/3] Starting TRL vLLM server on GPU ${SERVER_GPU} ..."
echo "      model=$MODEL_PATH"
echo "      host=$SERVER_HOST port=$SERVER_PORT"
echo "      gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION tp=$VLLM_TENSOR_PARALLEL_SIZE max_model_len=$VLLM_MAX_MODEL_LEN"
echo "      vllm_enforce_eager=$VLLM_ENFORCE_EAGER"
echo "      nccl: P2P=$NCCL_P2P_DISABLE IB=$NCCL_IB_DISABLE IFACE=$NCCL_SOCKET_IFNAME BLOCKING_WAIT=$TORCH_NCCL_BLOCKING_WAIT group_port=$TRL_VLLM_GROUP_PORT skip_barrier=$TRL_VLLM_SKIP_BARRIER sync_trace=$TRL_VLLM_SYNC_TRACE sync_barrier_interval=$TRL_VLLM_SYNC_BARRIER_INTERVAL trainable_only=$TRL_VLLM_SYNC_TRAINABLE_ONLY"

EAGER_FLAG="--enforce_eager"
if [[ "$VLLM_ENFORCE_EAGER" == "0" ]]; then
  EAGER_FLAG="--no-enforce_eager"
fi

# Clear stale log from previous runs to avoid false-positive failure detection.
: > /tmp/stage1_openr1_vllm_server.log

CUDA_VISIBLE_DEVICES="${SERVER_GPU}" setsid python "$ROOT/stage1/launch_trl_vllm_server_compat.py" \
  --model "$MODEL_PATH" \
  --host "$SERVER_HOST" \
  --port "$SERVER_PORT" \
  --tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max_model_len "$VLLM_MAX_MODEL_LEN" \
  "$EAGER_FLAG" \
  >/tmp/stage1_openr1_vllm_server.log 2>&1 &
VLLM_PID=$!

cleanup() {
  # Kill the whole process group to avoid leaving EngineCore worker processes behind.
  kill -- "-$VLLM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[2/3] Waiting for vLLM server to become healthy ..."
for i in $(seq 1 90); do
  if grep -qE "EngineCore failed to start|Engine core initialization failed|ImportError: cannot import name 'GuidedDecodingParams'|Process Process-1:|address already in use|Errno 98|duplicate template name" /tmp/stage1_openr1_vllm_server.log 2>/dev/null; then
    echo "Detected vLLM startup failure. Check /tmp/stage1_openr1_vllm_server.log"
    exit 1
  fi
  if curl -sf "http://${SERVER_HOST}:${SERVER_PORT}/get_world_size/" >/dev/null 2>&1 || \
     curl -sf "http://${SERVER_HOST}:${SERVER_PORT}/health/" >/dev/null 2>&1; then
    echo "TRL vLLM server is ready: http://${SERVER_HOST}:${SERVER_PORT}"
    break
  fi
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "vLLM server exited early. Check /tmp/stage1_openr1_vllm_server.log"
    exit 1
  fi
  if [[ "$i" == "90" ]]; then
    echo "Timed out waiting for vLLM server. Check /tmp/stage1_openr1_vllm_server.log"
    exit 1
  fi
  sleep 2
done

echo "[3/3] Launching Open-R1 GRPO training on GPU ${TRAIN_GPU} ..."
cd "$OPENR1_ROOT"
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL}" TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY}" \
  accelerate launch --config_file recipes/accelerate_configs/ddp.yaml --num_processes=1 \
  src/open_r1/grpo.py --config "$CFG" \
  --vllm_mode server \
  --vllm_server_base_url "http://${SERVER_HOST}:${SERVER_PORT}" \
  --vllm_server_host "$SERVER_HOST" \
  --vllm_server_port "$SERVER_PORT" \
  "$@"
