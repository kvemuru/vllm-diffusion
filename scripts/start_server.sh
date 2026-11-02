#!/usr/bin/env bash
#
# Start the vLLM-Omni OpenAI-compatible server for a Stable Diffusion model.
#
# All settings are environment variables so the same script works on a local
# host and inside the container. Any arguments passed to this script are
# forwarded to `vllm-omni serve` (for example: --enable-cpu-offload).
#
# vLLM-Omni 0.32.0 installs its own `vllm-omni` console script, which
# intercepts `--omni` and registers the diffusion flags. The plain `vllm`
# command from upstream vLLM does not know about `--omni`.
#
set -euo pipefail

VLLM_BIN="${VLLM_BIN:-vllm-omni}"
MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8091}"
TP_SIZE="${TP_SIZE:-1}"
VAE_USE_SLICING="${VAE_USE_SLICING:-1}"
VAE_USE_TILING="${VAE_USE_TILING:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

if ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
  echo "ERROR: '${VLLM_BIN}' was not found on PATH." >&2
  echo "Activate the environment and install 'vllm-omni==0.32.0' (which" >&2
  echo "provides the vllm-omni command), or set VLLM_BIN explicitly." >&2
  exit 1
fi

args=(
  serve "${MODEL_ID}"
  --omni
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
)

if [[ "${VAE_USE_SLICING}" == "1" ]]; then
  args+=(--vae-use-slicing)
fi

if [[ "${VAE_USE_TILING}" == "1" ]]; then
  args+=(--vae-use-tiling)
fi

# Intentional word splitting: EXTRA_VLLM_ARGS is a whitespace-separated list.
# shellcheck disable=SC2206
if [[ -n "${EXTRA_VLLM_ARGS}" ]]; then
  args+=(${EXTRA_VLLM_ARGS})
fi

# Allow callers to append/override flags, e.g. `docker compose run ... -- port`.
args+=("$@")

echo "Starting vLLM-Omni" >&2
echo "  model:  ${MODEL_ID}" >&2
echo "  listen: http://${HOST}:${PORT}" >&2
echo "  extra:  ${EXTRA_VLLM_ARGS:-<none>}" >&2
echo "  command: ${VLLM_BIN} ${args[*]}" >&2

exec "${VLLM_BIN}" "${args[@]}"
