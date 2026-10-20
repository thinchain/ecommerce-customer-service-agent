#!/usr/bin/env bash
# Shared offline serving environment, sourced by model launchers.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONFIG_PYTHON="${MODEL_CONFIG_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$CONFIG_PYTHON" ]]; then
  echo 'Create the project .venv with backend requirements, or set MODEL_CONFIG_PYTHON.' >&2
  exit 1
fi
# Parse first so VLLM_ENV itself can come from dotenv. Shell > private profile > .env.
# A checked substitution propagates parsing errors; no eval/source of dotenv text.
model_settings="$("$CONFIG_PYTHON" "$PROJECT_ROOT/scripts/model_settings.py" \
  "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.model.local")"
while IFS= read -r setting; do
  [[ -z "$setting" ]] && continue
  key="${setting%%=*}"
  value="${setting#*=}"
  if [[ ! -v "$key" ]]; then export "$key=$value"; fi
done <<< "$model_settings"
unset model_settings setting key value
: "${VLLM_ENV:?Set VLLM_ENV in .env.model.local, .env or the shell.}"
export VLLM_ENV
VLLM_PYTHON="$VLLM_ENV/bin/python"
if [[ ! -x "$VLLM_PYTHON" ]]; then echo 'VLLM_ENV must contain an executable bin/python.' >&2; exit 1; fi
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="$VLLM_ENV/bin:$PATH"
SITE_PACKAGES="$("$VLLM_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
LIBRARY_PATHS=("$VLLM_ENV/lib" "$SITE_PACKAGES/torch/lib")
for path in "$SITE_PACKAGES"/nvidia/*/lib; do
  if [[ -d "$path" ]]; then LIBRARY_PATHS+=("$path"); fi
done
if [[ -d "$VLLM_ENV/cuda-compat" ]]; then
  export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
  export VLLM_CUDA_COMPATIBILITY_PATH="$VLLM_ENV/cuda-compat"
  LIBRARY_PATHS=("$VLLM_ENV/cuda-compat" "${LIBRARY_PATHS[@]}")
fi
export LD_LIBRARY_PATH="$(IFS=:; printf '%s' "${LIBRARY_PATHS[*]}")"
check_model_port() {
  "$VLLM_PYTHON" - "$1" <<'PY'
import socket, sys
try:
    port = int(sys.argv[1])
    if not 1 <= port <= 65535: raise ValueError()
    with socket.socket() as sock: sock.bind(('127.0.0.1', port))
except (ValueError, OSError) as exc:
    sys.exit(f'Cannot use model port {sys.argv[1]}: {exc}. Reuse an existing service or choose another port.')
PY
}
