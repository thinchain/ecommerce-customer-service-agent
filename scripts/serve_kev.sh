#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/model_env.sh"
KEV_PORT="${KEV_PORT:-8795}"
: "${KEV_CHECKPOINT_PATH:?Set KEV_CHECKPOINT_PATH to your exported checkpoint.}"
: "${CUDA_VISIBLE_DEVICES:?Select an authorized GPU explicitly in the shell or local configuration.}"
export CUDA_VISIBLE_DEVICES
check_model_port "$KEV_PORT"
if [[ ! -f "$KEV_CHECKPOINT_PATH/decision_manifest.json" ]]; then
  echo "Kev checkpoint missing. Run: bash $PROJECT_ROOT/scripts/prepare_kev.sh" >&2; exit 1
fi
export VLLM_JEV_MAX_INFLIGHT="${VLLM_JEV_MAX_INFLIGHT:-8}"
export -n VLLM_ENV VLLM_PORT VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS
exec "$VLLM_ENV/bin/vllm-jev" serve "$KEV_CHECKPOINT_PATH" \
  --served-model-name "${KEV_MODEL:-kev-4b}" --host 127.0.0.1 --port "$KEV_PORT" \
  --gpu-memory-utilization "${KEV_GPU_MEMORY_UTILIZATION:-0.30}" \
  --max-model-len "${KEV_MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${KEV_MAX_NUM_SEQS:-4}" --enforce-eager
