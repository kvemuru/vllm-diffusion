#!/usr/bin/env bash
#
# Run the diffusion server with the native Podman CLI (no Compose needed).
#
#   cp .env.example .env
#   ./scripts/podman_run.sh
#
# GPU access uses CDI by default. Generate a CDI spec once per host:
#   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml        # rootful
#   nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml    # rootless
#
# Override GPU_ARGS if your host uses a different mechanism, for example:
#   GPU_ARGS="--device /dev/nvidia0 --device /dev/nvidiactl" ./scripts/podman_run.sh
#   GPU_ARGS="--gpus all" ./scripts/podman_run.sh        # if your Podman supports it
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Load local configuration if present. Compose does this automatically; the
# Podman CLI does not.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

IMAGE="${IMAGE:-sd-vllm:0.32.0}"
CONTAINER_NAME="${CONTAINER_NAME:-sd-vllm-server}"
MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
HOST="${HOST:-0.0.0.0}"
BIND_ADDRESS="${BIND_ADDRESS:-0.0.0.0}"
PORT="${PORT:-8091}"
TP_SIZE="${TP_SIZE:-1}"
VAE_USE_SLICING="${VAE_USE_SLICING:-1}"
VAE_USE_TILING="${VAE_USE_TILING:-1}"
HF_TOKEN="${HF_TOKEN:-}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
VLLM_BIN="${VLLM_BIN:-vllm-omni}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
VOLUME_NAME="${VOLUME_NAME:-sd-vllm-hf-cache}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${PWD}/outputs}"
OUTPUTS_VOLUME_OPT="${OUTPUTS_VOLUME_OPT:-}"
GPU_ARGS="${GPU_ARGS:---device nvidia.com/gpu=all}"

if ! command -v podman >/dev/null 2>&1; then
  echo "ERROR: podman was not found on PATH." >&2
  exit 1
fi

mkdir -p "${OUTPUTS_DIR}"
if ! podman volume exists "${VOLUME_NAME}"; then
  podman volume create "${VOLUME_NAME}" >/dev/null
  echo "Created Podman volume ${VOLUME_NAME}."
fi

# Intentional word splitting so GPU_ARGS can contain several flags.
# shellcheck disable=SC2206
gpu_args=(${GPU_ARGS})

args=(
  run
  --detach
  --replace
  --name "${CONTAINER_NAME}"
  --ipc=host
  "${gpu_args[@]}"
  --publish "${BIND_ADDRESS}:${PORT}:${PORT}"
  --env "MODEL_ID=${MODEL_ID}"
  --env "HOST=${HOST}"
  --env "PORT=${PORT}"
  --env "TP_SIZE=${TP_SIZE}"
  --env "VAE_USE_SLICING=${VAE_USE_SLICING}"
  --env "VAE_USE_TILING=${VAE_USE_TILING}"
  --env "HF_HOME=/root/.cache/huggingface"
  --env "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
  --env "VLLM_BIN=${VLLM_BIN}"
  --env "EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS}"
  --volume "${VOLUME_NAME}:/root/.cache/huggingface"
  --volume "${OUTPUTS_DIR}:/app/outputs${OUTPUTS_VOLUME_OPT}"
  --restart=unless-stopped
)

if [[ -n "${HF_TOKEN}" ]]; then
  args+=(--env "HF_TOKEN=${HF_TOKEN}")
fi

args+=("${IMAGE}" "$@")

echo "Starting ${CONTAINER_NAME} from ${IMAGE}..."
echo "  model:  ${MODEL_ID}"
echo "  listen: ${BIND_ADDRESS}:${PORT}"
echo "  GPU:    ${GPU_ARGS}"
echo "  volume: ${VOLUME_NAME} -> /root/.cache/huggingface"

podman "${args[@]}"

echo
echo "Container started. Useful commands:"
echo "  podman logs -f ${CONTAINER_NAME}"
echo "  curl -fsS http://127.0.0.1:${PORT}/health"
echo "  podman rm -f ${CONTAINER_NAME}"
