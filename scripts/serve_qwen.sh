#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/model_env.sh"
: "${MODEL_PATH:?Set MODEL_PATH to your local chat checkpoint.}"
: "${CUDA_VISIBLE_DEVICES:?Select an authorized GPU explicitly in the shell or local configuration.}"
export CUDA_VISIBLE_DEVICES
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1536M}"
MODEL_GPU_MEMORY_UTILIZATION="${MODEL_GPU_MEMORY_UTILIZATION:-0.40}"
check_model_port "$VLLM_PORT"
mapfile -t profile < <("$VLLM_PYTHON" - "$MODEL_PATH" <<'PY_PROFILE'
import json, sys
from pathlib import Path
config = json.loads((Path(sys.argv[1]) / 'config.json').read_text())
kind = config['model_type']
if kind in ('qwen3_5', 'qwen3_5_text'):
    print('qwen3.5-4b'); print('bfloat16'); print('qwen3_coder'); print('none'); print(kind)
elif kind == 'qwen3' and config.get('quantization_config', {}).get('quant_method') == 'awq':
    print('qwen3-8b-awq'); print('half'); print('hermes'); print('awq'); print(kind)
else:
    sys.exit(f'Unsupported chat model type: {kind}')
PY_PROFILE
)
if [[ ${#profile[@]} -ne 5 ]]; then
  echo "Could not identify chat checkpoint: $MODEL_PATH" >&2; exit 1
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${QWEN_MODEL:-${profile[0]}}}"
args=(serve "$MODEL_PATH" --host 127.0.0.1 --port "$VLLM_PORT"
  --served-model-name "$SERVED_MODEL_NAME" --dtype "${profile[1]}"
  --max-model-len "$VLLM_MAX_MODEL_LEN" --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --gpu-memory-utilization "$MODEL_GPU_MEMORY_UTILIZATION"
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
  --enable-auto-tool-choice --tool-call-parser "${profile[2]}"
  --enforce-eager --api-key "${VLLM_API_KEY:-${OPENAI_API_KEY:-local-dev-key}}")
if [[ ${profile[3]} != none ]]; then args+=(--quantization "${profile[3]}"); fi
if [[ ${profile[4]} == qwen3_5 ]]; then args+=(--language-model-only); fi
printf 'Chat server: GPU=%s, model=%s, port=%s, parser=%s\n' "$CUDA_VISIBLE_DEVICES" "$SERVED_MODEL_NAME" "$VLLM_PORT" "${profile[2]}"
export -n VLLM_ENV VLLM_PORT VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS
exec "$VLLM_ENV/bin/vllm" "${args[@]}"
