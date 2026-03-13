#!/usr/bin/env bash
set -euo pipefail

# Wrapper for Stage2 training that reuses the Stage1-proven server runner.

ROOT="${ROOT:-/root/grpo}"
export ROOT
export OPENR1_ROOT="${OPENR1_ROOT:-$ROOT/open-r1}"
export CFG="${CFG:-$ROOT/stage2/openr1_stage2_grpo_server_2x5090.yaml}"

exec bash "$ROOT/stage1/run_stage1_openr1_server.sh" "$@"
