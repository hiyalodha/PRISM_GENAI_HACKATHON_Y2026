#!/usr/bin/env bash
set -euo pipefail

MLX_VENV="${MLX_VENV:-.venv-mlx}"
MODEL="${PRISM_LOCAL_MODEL:-mlx-community/Qwen3-VL-8B-Instruct-4bit}"
HOST="${PRISM_LOCAL_HOST:-127.0.0.1}"
PORT="${PRISM_LOCAL_PORT:-8081}"

if [[ ! -x "${MLX_VENV}/bin/python" ]]; then
  echo "MLX environment not found at ${MLX_VENV}. Run 'make install-mlx' first." >&2
  exit 1
fi

echo "Serving ${MODEL} on http://${HOST}:${PORT}/v1"
exec "${MLX_VENV}/bin/python" -m mlx_vlm.server --model "${MODEL}" --host "${HOST}" --port "${PORT}"
