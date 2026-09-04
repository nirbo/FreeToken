#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${FREETOKEN_VENV:-/home/nir/venvs/freetoken-dev}"

export PYTHONPATH="$ROOT/python"
export HF_HOME="${HF_HOME:-/home/nir/venvs/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/nir/venvs/.cache/triton}"

exec "$VENV/bin/ft" serve \
  --model "${FREETOKEN_MODEL:-/mnt/fast/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP}" \
  --reasoning-effort medium \
  --speculative-tokens 4 \
  --max-running-requests 1 \
  --host 0.0.0.0 \
  --port 1234 \
  "$@"
