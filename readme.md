# Stable Diffusion Text-to-Image with vLLM-Omni

A single-host deployment of **Stable Diffusion XL 1.0** and **Stable Diffusion
3.5** behind an OpenAI-compatible image-generation API.

The server is **vLLM-Omni** (`vllm-project/vllm-omni`), the diffusion
extension of vLLM. Upstream vLLM only serves autoregressive language models.
vLLM-Omni adds diffusion pipelines and installs its own `vllm-omni` command:

```bash
vllm-omni serve <model> --omni
```

Do not use the upstream `vllm serve ... --omni` command; it does not register
the `--omni` flag.

## Validated stack

| Component | Version |
| --- | --- |
| vLLM-Omni | `0.32.0` |
| vLLM | `0.32.0` |
| Python | `3.12` |
| CUDA | `13.0` |
| PyTorch | `2.13.0+cu130` |
| NVIDIA driver | `580.x` or newer |

Supported text-to-image checkpoints:

- `stabilityai/stable-diffusion-xl-base-1.0` (default, open)
- `stabilityai/stable-diffusion-3.5-medium` (gated)
- `stabilityai/stable-diffusion-3.5-large` (gated)

For gated models, accept the license on Hugging Face and set `HF_TOKEN` in
`.env`.

## Repository layout

```text
.
├── Dockerfile                 # vLLM CUDA image + vLLM-Omni runtime
├── docker-compose.yaml        # API service, GPU reservation, optional UI
├── environment.yml            # Conda alternative for non-container hosts
├── Makefile                   # Docker Compose and Podman shortcuts
├── .env.example               # Configuration template
├── app/
│   └── gradio_app.py          # Gradio prompt UI (HTTP client)
├── scripts/
│   ├── start_server.sh        # Env vars -> vllm-omni serve
│   ├── podman_run.sh          # Native Podman runner (no Compose)
│   ├── generate.py            # Stdlib-only Python API client
│   └── smoke_test.sh          # Health wait + generate + PNG validation
└── outputs/                   # Generated images
```

How a request flows:

1. The server loads the checkpoint from the Hugging Face cache at startup.
2. A client posts a prompt to `POST /v1/images/generations`.
3. vLLM-Omni runs the text encoders, denoiser (UNet/MMDiT), and VAE decoder.
4. The API returns a base64 PNG or a file URL.

Only one model is served per process. Change `MODEL_ID` and restart to switch.

## Requirements

- Linux host with an NVIDIA GPU (compute capability 7.5+) and driver 580+.
- Container runtime:
  - Docker Engine 24+ with Compose v2.30+, or
  - Podman 4.9+ (rootless is supported).
- NVIDIA Container Toolkit.
- Free disk space: about 30 GB for the image, 7 GB for SDXL weights, 16 GB for
  SD 3.5 Medium, and 32 GB for SD 3.5 Large.
- Ports `8091` (API) and optionally `7860` (Gradio UI).

Check the driver:

```bash
nvidia-smi
```

The reported CUDA version must be `13.0` or newer.

## Build the container

### Docker Compose

```bash
cd /home/kvemuru/vllm-diffusion
cp .env.example .env
docker compose build
```

### Podman

```bash
cd /home/kvemuru/vllm-diffusion
cp .env.example .env
podman build -t sd-vllm:0.32.0 .
```

The build pulls the official `docker.io/vllm/vllm-openai:v0.32.0` base image
(about 9 GB compressed), installs `vllm-omni[demo]==0.32.0`, and produces an
image of about 25 GB.

The Dockerfile uses a fully qualified base image name so rootless Podman does
not need a short-name registry alias.

## Run the server

### Docker

```bash
cp .env.example .env
docker compose up -d
docker compose logs -f sd-server
```

### Podman

Podman accesses NVIDIA GPUs through CDI. Generate a CDI specification once:

```bash
# rootful
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# rootless
nvidia-ctk cdi generate --output="$HOME/.config/cdi/nvidia.yaml"
```

Verify GPU access:

```bash
podman run --rm --device nvidia.com/gpu=all \
  docker.io/nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

Then start the server:

```bash
cp .env.example .env
./scripts/podman_run.sh
# or
make podman-up
```

The runner sources `.env`, creates the `sd-vllm-hf-cache` volume on first use,
publishes `${PORT}`, mounts `./outputs`, and uses
`--device nvidia.com/gpu=all` by default.

The first start downloads the model weights, which can take several minutes.
The startup health grace period is 15 minutes.

## Test text-to-image generation

### 1. Check health and models

```bash
curl -fsS http://127.0.0.1:8091/health
curl -fsS http://127.0.0.1:8091/v1/models
```

### 2. Run the smoke test

The smoke test waits for the server, requests one image, and validates the PNG:

```bash
bash scripts/smoke_test.sh
# Docker: make smoke
# Podman: make podman-smoke
```

Output: `outputs/smoke-test.png`.

### 3. Generate with the Python client

The client uses only the Python standard library and runs on the host:

```bash
python3 scripts/generate.py \
  --prompt "a red fox in a snowy forest, golden hour, 85mm photo" \
  --negative-prompt "blurry, low quality, watermark" \
  --size 1024x1024 \
  --steps 30 \
  --guidance-scale 7.5 \
  --seed 42 \
  --n 2 \
  --output-dir outputs
```

Useful options: `--api-base`, `--model`, `--size`, `--steps`,
`--guidance-scale`, `--seed`, `--n`, `--timeout`.

### 4. Call the API directly

```bash
curl -fsS http://127.0.0.1:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "a lighthouse on a rocky coast at dusk, photorealistic",
    "negative_prompt": "blurry, low quality, watermark, text",
    "size": "1024x1024",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 42,
    "n": 1,
    "response_format": "b64_json"
  }'
```

### 5. Optional Gradio UI

```bash
make podman-ui
# or
docker compose --profile ui up -d
```

Open <http://127.0.0.1:7860>. The Podman target points the UI at
`http://127.0.0.1:8091`; Compose points it at `http://sd-server:8091`. The UI
supports prompt, negative prompt, size, steps, guidance, seed, and image count.

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | `stabilityai/stable-diffusion-xl-base-1.0` | Model repo id or local path. |
| `PORT` | `8091` | API port. |
| `HOST` | `0.0.0.0` | Server bind address inside the container. |
| `TP_SIZE` | `1` | Tensor parallelism for multi-GPU hosts. |
| `VAE_USE_SLICING` | `1` | Reduce VAE decode memory. |
| `VAE_USE_TILING` | `1` | Reduce VAE decode memory. |
| `EXTRA_VLLM_ARGS` | empty | Extra `vllm-omni serve` flags. |
| `VLLM_BIN` | `vllm-omni` | Console script used by the startup script. |
| `HF_TOKEN` | empty | Token for gated/private checkpoints. |
| `API_BASE` | `http://127.0.0.1:8091` | Used by the client and UI. |
| `GRADIO_SERVER_PORT` | `7860` | Gradio port. |
| `BIND_ADDRESS` | `0.0.0.0` | Podman host publish address. |
| `GPU_ARGS` | `--device nvidia.com/gpu=all` | Podman GPU flags. |

Example for a smaller GPU:

```dotenv
EXTRA_VLLM_ARGS=--enable-cpu-offload --disable-log-requests
```

To switch models, edit `.env` and recreate the server:

```dotenv
MODEL_ID=stabilityai/stable-diffusion-3.5-medium
```

## Stop and clean up

```bash
# Docker
docker compose down          # keep the model cache
docker compose down -v       # delete the model cache too

# Podman
make podman-down
podman volume rm sd-vllm-hf-cache
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `unrecognized arguments: --omni` | Use `vllm-omni serve ... --omni`, not upstream `vllm`. The provided script does this. |
| `401`/`403` while downloading | Accept the model license on Hugging Face and set `HF_TOKEN`. |
| `CUDA driver version is insufficient` | Upgrade to NVIDIA driver 580.x or newer. |
| CUDA out of memory | Lower size/steps, keep VAE slicing/tiling, add `--enable-cpu-offload`. |
| Podman: `nvidia.com/gpu=all: no such device` | Generate the CDI spec with `nvidia-ctk cdi generate`. |
| Server stays unhealthy | The first run is downloading weights. Check `docker compose logs -f sd-server` or `podman logs -f sd-vllm-server`. |
| `curl: (7) Failed to connect` | The server is not running or `PORT` is wrong. Check `docker compose ps` or `podman ps -a`. |

## Validation status

Validated on the authoring host (Linux x86_64, rootless Podman 4.9.3, no NVIDIA
GPU):

- `podman build -t sd-vllm:0.32.0 .` completes; image size is about 25.1 GB.
- The image contains Python 3.12.3, torch 2.13.0+cu130, vllm 0.32.0,
  vllm_omni 0.32.0, and Gradio 6.31.0.
- `vllm-omni serve ... --omni --help=all` exits 0 and lists `--omni`,
  `--tensor-parallel-size`, `--vae-use-slicing`, `--vae-use-tiling`, and
  `--enable-cpu-offload`.
- Shell syntax, Python compilation, and YAML parsing pass.
- `generate.py`, `smoke_test.sh`, `start_server.sh`, `podman_run.sh`, and the
  Gradio handlers pass an offline harness against a mock OpenAI-compatible
  server, including error paths and PNG output validation.

Not executed here: real model download and GPU inference (no GPU on the
authoring host), and Docker Compose runtime (no Docker daemon or Compose
provider). Run `make podman-smoke` or `make smoke` on the target GPU host to
validate the end-to-end model path.

## References

- vLLM-Omni: <https://github.com/vllm-project/vllm-omni>
- Installation: <https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/>
- Image generation API: <https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/image_generation_api>
- SDXL recipe: <https://github.com/vllm-project/vllm-omni/blob/main/recipes/StabilityAI/Stable-Diffusion-XL.md>
- SD 3.5 recipe: <https://github.com/vllm-project/vllm-omni/blob/main/recipes/StabilityAI/Stable-Diffusion-3.5.md>
- Docker images: <https://hub.docker.com/r/vllm/vllm-openai>
- Podman: <https://docs.podman.io/>
- NVIDIA CDI: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html>
