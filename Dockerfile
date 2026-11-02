# syntax=docker/dockerfile:1
#
# Stable Diffusion text-to-image serving with vLLM-Omni.
#
# vLLM-Omni extends vLLM with diffusion pipelines and exposes them through the
# OpenAI-compatible `vllm-omni serve ... --omni` command. The official CUDA
# image already contains vLLM 0.32.0, CUDA 13.0 and PyTorch 2.13.0, so this
# image only adds the vLLM-Omni diffusion runtime and the demo/client tools.
#
# Build:
#   docker build -t sd-vllm:0.32.0 .
#
# Run:
#   docker run --gpus all --ipc=host -p 8091:8091 \
#     -e MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0 \
#     -v sd-hf-cache:/root/.cache/huggingface \
#     sd-vllm:0.32.0

# Fully qualified so both Docker and rootless Podman resolve the image.
FROM docker.io/vllm/vllm-openai:v0.32.0

ARG VLLM_OMNI_VERSION=0.32.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0 \
    HOST=0.0.0.0 \
    PORT=8091 \
    TP_SIZE=1 \
    VAE_USE_SLICING=1 \
    VAE_USE_TILING=1

# curl is used by the container health check and the smoke-test helper.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install the diffusion/omni runtime. `[demo]` adds Gradio and the client
# dependencies used by app/gradio_app.py. Versions are pinned to the matching
# vLLM 0.32.0 release for reproducibility.
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install \
        "vllm-omni[demo]==${VLLM_OMNI_VERSION}" \
        "huggingface-hub[hf-transfer]>=0.34" \
        "pillow>=11.0"

WORKDIR /app

COPY scripts ./scripts
COPY app ./app
COPY .env.example ./.env.example

RUN chmod +x ./scripts/*.sh \
    && mkdir -p /app/outputs

# Model cache and generated images. Use a named volume or bind mount for both
# so images survive container replacement.
VOLUME ["/root/.cache/huggingface", "/app/outputs"]

EXPOSE 8091 7860

# First start downloads several GB of weights, so allow a long start period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=20 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8091}/health" || exit 1

# The entrypoint reads MODEL_ID/PORT/TP_SIZE/VAE_* from the environment.
# Extra `docker run` arguments are forwarded to `vllm-omni serve`.
ENTRYPOINT ["/app/scripts/start_server.sh"]
