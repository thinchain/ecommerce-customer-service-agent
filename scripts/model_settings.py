"""Emit whitelisted, literal model settings for Bash. Never execute dotenv text."""
import sys

from dotenv import dotenv_values

ALLOWED = {
    'VLLM_ENV', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'MODEL_PATH', 'QWEN_MODEL',
    'OPENAI_API_KEY', 'VLLM_API_KEY', 'SERVED_MODEL_NAME', 'KV_CACHE_MEMORY_BYTES',
    'VLLM_PORT', 'VLLM_MAX_NUM_SEQS', 'VLLM_MAX_MODEL_LEN', 'MODEL_GPU_MEMORY_UTILIZATION',
    'KEV_SOURCE_PATH', 'KEV_BASE_PATH', 'KEV_CHECKPOINT_PATH', 'KEV_PORT', 'KEV_MODEL',
    'KEV_GPU_MEMORY_UTILIZATION', 'KEV_MAX_MODEL_LEN', 'KEV_MAX_NUM_SEQS',
}


def load_settings(paths):
    values = {}
    for path in paths:
        for key, value in dotenv_values(path, interpolate=False).items():
            if key in ALLOWED and value is not None:
                if any(char in value for char in ('\n', '\r', '\0')):
                    raise ValueError(f'{key}: multiline values are not supported by model launchers')
                values[key] = value
    return values


def main():
    try:
        values = load_settings(sys.argv[1:])
    except (ValueError, OSError) as exc:
        print(f'Model configuration error: {exc}', file=sys.stderr)
        return 1
    for key, value in values.items():
        print(f'{key}={value}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
