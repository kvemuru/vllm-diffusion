SHELL := /bin/bash
COMPOSE ?= docker compose
PODMAN ?= podman
IMAGE ?= sd-vllm:0.32.0
CONTAINER_NAME ?= sd-vllm-server
VOLUME_NAME ?= sd-vllm-hf-cache

# Pick up MODEL_ID/PORT/etc. from .env when present. Compose also reads this
# file directly; this keeps the Make targets in sync with the served model.
-include .env

PORT ?= 8091
MODEL ?= $(if $(MODEL_ID),$(MODEL_ID),stabilityai/stable-diffusion-xl-base-1.0)
PROMPT ?= a red fox in a snowy forest, golden hour, 85mm photo
OUTPUT ?= outputs/smoke-test.png

.PHONY: help build up down restart logs ps smoke generate ui shell download-model clean \
	podman-build podman-up podman-down podman-logs podman-smoke podman-ui podman-shell \
	podman-download-model podman-clean

help:
	@echo "Targets:"
	@echo "  make build          Build the container image (Docker Compose)"
	@echo "  make up             Start the diffusion server in the background"
	@echo "  make down           Stop and remove the containers"
	@echo "  make restart        Restart the diffusion server"
	@echo "  make logs           Follow server logs"
	@echo "  make ps             Show container status"
	@echo "  make smoke          Wait for health and generate a test image"
	@echo "  make generate       Generate an image (PROMPT=... OUTPUT=...)"
	@echo "  make ui             Start the Gradio UI on port 7860"
	@echo "  make download-model Pre-download MODEL into the Docker HF cache volume"
	@echo "  make shell          Open a shell inside the server container"
	@echo "  make clean          Stop containers and delete local outputs"
	@echo ""
	@echo "Podman CLI (no Compose required):"
	@echo "  make podman-build   Build $(IMAGE)"
	@echo "  make podman-up      Start the server with scripts/podman_run.sh"
	@echo "  make podman-down    Remove the server container"
	@echo "  make podman-logs    Follow server logs"
	@echo "  make podman-smoke   Wait for health and generate a test image"
	@echo "  make podman-ui      Start the Gradio UI with host networking"
	@echo "  make podman-shell   Open a shell in the server container"
	@echo "  make podman-download-model  Pre-download MODEL into the Podman volume"
	@echo "  make podman-clean   Stop/remove the container and delete local outputs"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart sd-server

logs:
	$(COMPOSE) logs -f sd-server

ps:
	$(COMPOSE) ps

smoke:
	API_BASE=http://127.0.0.1:$(PORT) MODEL_ID=$(MODEL) bash scripts/smoke_test.sh $(OUTPUT)

generate:
	python3 scripts/generate.py \
		--api-base http://127.0.0.1:$(PORT) \
		--model $(MODEL) \
		--prompt "$(PROMPT)" \
		--output-dir outputs

ui:
	$(COMPOSE) --profile ui up -d sd-ui
	@echo "Gradio UI: http://127.0.0.1:7860"

download-model:
	$(COMPOSE) run --rm --entrypoint hf sd-server download $(MODEL)

shell:
	$(COMPOSE) exec sd-server bash

clean:
	$(COMPOSE) down
	rm -f outputs/*.png outputs/*.jpg outputs/*.jpeg outputs/*.webp

# --- Native Podman CLI (no Compose plugin needed) ---------------------------

podman-build:
	$(PODMAN) build -t $(IMAGE) .

podman-up:
	IMAGE=$(IMAGE) CONTAINER_NAME=$(CONTAINER_NAME) VOLUME_NAME=$(VOLUME_NAME) \
		bash scripts/podman_run.sh

podman-down:
	-$(PODMAN) rm -f $(CONTAINER_NAME)

podman-logs:
	$(PODMAN) logs -f $(CONTAINER_NAME)

podman-smoke:
	API_BASE=http://127.0.0.1:$(PORT) MODEL_ID=$(MODEL) bash scripts/smoke_test.sh $(OUTPUT)

podman-ui:
	-$(PODMAN) rm -f sd-vllm-ui
	$(PODMAN) run -d --name sd-vllm-ui --network=host \
		-e API_BASE=http://127.0.0.1:$(PORT) \
		-e MODEL_ID=$(MODEL) \
		-e GRADIO_SERVER_NAME=0.0.0.0 \
		-e GRADIO_SERVER_PORT=7860 \
		--entrypoint python3 \
		$(IMAGE) /app/app/gradio_app.py
	@echo "Gradio UI: http://127.0.0.1:7860"

podman-shell:
	$(PODMAN) exec -it $(CONTAINER_NAME) bash

podman-download-model:
	$(PODMAN) run --rm --entrypoint hf \
		-e HF_TOKEN=$(HF_TOKEN) \
		-v $(VOLUME_NAME):/root/.cache/huggingface \
		$(IMAGE) download $(MODEL)

podman-clean:
	-$(PODMAN) rm -f $(CONTAINER_NAME) sd-vllm-ui
	rm -f outputs/*.png outputs/*.jpg outputs/*.jpeg outputs/*.webp
