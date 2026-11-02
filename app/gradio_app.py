#!/usr/bin/env python3
"""Minimal Gradio front end for a local vLLM-Omni Stable Diffusion server.

The UI is only an HTTP client; it does not load the diffusion model itself.

Environment:
    API_BASE            vLLM-Omni base URL (default http://127.0.0.1:8091)
    MODEL_ID            served model id (default stabilityai/stable-diffusion-xl-base-1.0)
    REQUEST_TIMEOUT     per-request timeout in seconds (default 900)
    GRADIO_SERVER_NAME  bind address (default 0.0.0.0)
    GRADIO_SERVER_PORT  port (default 7860)
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import time

import gradio as gr
import requests
from PIL import Image

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8091").rstrip("/")
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "900"))

SIZE_CHOICES = [
    "512x512",
    "768x768",
    "1024x1024",
    "1152x896",
    "896x1152",
    "1024x768",
    "768x1024",
    "1344x768",
    "768x1344",
]


def server_status() -> str:
    """Return a short status line for the diffusion server."""
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5)
        if health.status_code != 200:
            return f"Not ready: GET /health returned HTTP {health.status_code}"
        models = requests.get(f"{API_BASE}/v1/models", timeout=10)
        model_ids = [item.get("id") for item in models.json().get("data", [])] if models.ok else []
        served = ", ".join(model_ids) if model_ids else MODEL_ID
        return f"Ready at {API_BASE}\nServed model: {served}"
    except requests.RequestException as exc:
        return f"Unavailable at {API_BASE}\n{exc}"


def generate(
    prompt: str,
    negative_prompt: str,
    size: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    batch: int,
) -> tuple[list[Image.Image], str, int]:
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    actual_seed = random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)
    payload = {
        "model": MODEL_ID,
        "prompt": prompt.strip(),
        "size": size,
        "n": int(batch),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "seed": actual_seed,
        "response_format": "b64_json",
    }
    if negative_prompt and negative_prompt.strip():
        payload["negative_prompt"] = negative_prompt.strip()

    started = time.perf_counter()
    try:
        response = requests.post(
            f"{API_BASE}/v1/images/generations",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise gr.Error(f"Could not reach {API_BASE}: {exc}") from exc

    if response.status_code >= 400:
        raise gr.Error(f"HTTP {response.status_code}: {response.text[:1500]}")

    images: list[Image.Image] = []
    for index, item in enumerate(response.json().get("data", [])):
        try:
            if item.get("b64_json"):
                raw = base64.b64decode(item["b64_json"])
            elif item.get("url"):
                raw = requests.get(item["url"], timeout=REQUEST_TIMEOUT).content
            else:
                raise ValueError(f"item {index} has no b64_json/url")
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except (ValueError, KeyError, OSError) as exc:
            raise gr.Error(f"Could not decode image {index}: {exc}") from exc

    if not images:
        raise gr.Error("The server returned no images.")

    elapsed = time.perf_counter() - started
    metadata = {
        "model": MODEL_ID,
        "prompt": payload["prompt"],
        "negative_prompt": payload.get("negative_prompt"),
        "size": size,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "seed": actual_seed,
        "n": int(batch),
        "elapsed_seconds": round(elapsed, 2),
    }
    return images, json.dumps(metadata, indent=2), actual_seed


with gr.Blocks(title="Stable Diffusion + vLLM-Omni") as demo:
    gr.Markdown(
        "# Stable Diffusion Text-to-Image\n"
        f"Server: `{API_BASE}` &nbsp;|&nbsp; Model: `{MODEL_ID}`"
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="a red fox in a snowy forest, golden hour, 85mm photo",
                lines=4,
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                placeholder="blurry, low quality, watermark, text",
                lines=2,
            )
            with gr.Accordion("Generation settings", open=True):
                size = gr.Dropdown(choices=SIZE_CHOICES, value="1024x1024", label="Size")
                with gr.Row():
                    steps = gr.Slider(1, 60, value=30, step=1, label="Inference steps")
                    guidance = gr.Slider(0.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    batch = gr.Slider(1, 4, value=1, step=1, label="Images")

            with gr.Row():
                run = gr.Button("Generate", variant="primary", scale=3)
                check = gr.Button("Check server", scale=1)

            status = gr.Textbox(label="Server status", interactive=False, lines=2)
            metadata = gr.Textbox(label="Last request", interactive=False, lines=9)

        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Result", columns=2, height=640, object_fit="contain")
            used_seed = gr.Number(label="Seed used", precision=0, interactive=False)

    gr.Examples(
        examples=[
            [
                "a red fox in a snowy forest, golden hour, 85mm photo",
                "blurry, low quality, watermark, text",
                "1024x1024",
                30,
                7.5,
                42,
                1,
            ],
            [
                "a ceramic teapot on a wooden table, soft window light, product photo",
                "distorted, extra objects, text",
                "1024x1024",
                28,
                6.5,
                -1,
                2,
            ],
            [
                "isometric miniature city at night, neon signs, rain-slick streets",
                "flat, washed out, low detail",
                "1152x896",
                35,
                8.0,
                1234,
                1,
            ],
        ],
        inputs=[prompt, negative_prompt, size, steps, guidance, seed, batch],
    )

    run.click(
        generate,
        inputs=[prompt, negative_prompt, size, steps, guidance, seed, batch],
        outputs=[gallery, metadata, used_seed],
    )
    check.click(server_status, outputs=status)

    demo.load(server_status, outputs=status)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
