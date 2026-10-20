#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/model_env.sh"
: "${KEV_SOURCE_PATH:?Set KEV_SOURCE_PATH to your local adapter.}"
: "${KEV_BASE_PATH:?Set KEV_BASE_PATH to your local base checkpoint.}"
: "${KEV_CHECKPOINT_PATH:?Set KEV_CHECKPOINT_PATH to a new export directory.}"
CUDA_VISIBLE_DEVICES='' "$VLLM_PYTHON" "$PROJECT_ROOT/scripts/prepare_kev.py" \
  --source "$KEV_SOURCE_PATH" \
  --base "$KEV_BASE_PATH" \
  --output "$KEV_CHECKPOINT_PATH"
