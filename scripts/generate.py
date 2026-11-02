#!/usr/bin/env python3
"""Generate images from a running vLLM-Omni diffusion server.

This client uses only the Python standard library, so it works on the host
without installing the model dependencies.

Examples:
    python3 scripts/generate.py \
        --prompt "a red fox in a snowy forest, golden hour, 85mm photo" \
        --negative-prompt "blurry, low quality, watermark" \
        --seed 42

    python3 scripts/generate.py --prompt "a teapot" --size 768x768 \
        --steps 25 --guidance-scale 6.0 --n 2 --output-dir outputs
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib import error, request

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_API_BASE = "http://127.0.0.1:8091"
SIZE_RE = re.compile(r"^(\d{3,4})x(\d{3,4})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the vLLM-Omni /v1/images/generations endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", required=True, help="Text description of the image.")
    parser.add_argument("--negative-prompt", default=None, help="What to avoid (needs guidance_scale > 1).")
    parser.add_argument("--api-base", default=os.getenv("API_BASE", DEFAULT_API_BASE), help="Server base URL.")
    parser.add_argument("--model", default=os.getenv("MODEL_ID", DEFAULT_MODEL), help="Served model id.")
    parser.add_argument("--size", default="1024x1024", help="Image size as WIDTHxHEIGHT.")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--n", type=int, default=1, help="Number of images to generate (1-10).")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated PNG files.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout in seconds.")
    return parser.parse_args()


def validate_size(size: str) -> tuple[int, int]:
    match = SIZE_RE.match(size.strip().lower())
    if not match:
        raise SystemExit(f"Invalid --size {size!r}; expected WIDTHxHEIGHT, e.g. 1024x1024")

    width, height = (int(value) for value in match.groups())
    if width % 8 != 0 or height % 8 != 0:
        raise SystemExit("Width and height must be multiples of 8.")
    if not (256 <= width <= 2048 and 256 <= height <= 2048):
        raise SystemExit("Width and height must be between 256 and 2048.")
    return width, height


def post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}:\n{detail}") from exc
    except error.URLError as exc:
        raise SystemExit(
            f"Could not reach {url}: {exc.reason}\n"
            "Is the server running? Try: docker compose up -d && make smoke"
        ) from exc


def fetch_url(url: str, timeout: float) -> bytes:
    with request.urlopen(url, timeout=timeout) as response:
        return response.read()


def save_images(data: list[dict], output_dir: Path, seed: int | None, timeout: float) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    saved: list[Path] = []

    for index, item in enumerate(data):
        if item.get("b64_json"):
            raw = base64.b64decode(item["b64_json"])
        elif item.get("url"):
            raw = fetch_url(item["url"], timeout)
        else:
            raise SystemExit(f"Response item {index} has neither b64_json nor url: {item!r}")

        suffix = f"_seed{seed}" if seed is not None else ""
        path = output_dir / f"sd_{stamp}_{index}{suffix}.png"
        path.write_bytes(raw)
        saved.append(path)

    return saved


def main() -> int:
    args = parse_args()
    if not args.prompt.strip():
        raise SystemExit("--prompt must not be empty.")
    if not 1 <= args.n <= 10:
        raise SystemExit("--n must be between 1 and 10.")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1.")

    width, height = validate_size(args.size)

    payload: dict[str, object] = {
        "model": args.model,
        "prompt": args.prompt,
        "size": f"{width}x{height}",
        "n": args.n,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "response_format": "b64_json",
    }
    if args.negative_prompt:
        payload["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        payload["seed"] = args.seed

    endpoint = f"{args.api_base.rstrip('/')}/v1/images/generations"
    started = time.perf_counter()
    result = post_json(endpoint, payload, args.timeout)
    elapsed = time.perf_counter() - started

    items = result.get("data") or []
    if not items:
        raise SystemExit(f"Server returned no images:\n{json.dumps(result, indent=2)[:2000]}")

    paths = save_images(items, Path(args.output_dir), args.seed, args.timeout)

    print(f"Generated {len(paths)} image(s) in {elapsed:.1f}s")
    print(f"  model: {args.model}")
    print(f"  size:  {width}x{height}, steps={args.steps}, guidance={args.guidance_scale}")
    for path in paths:
        print(f"  saved: {path} ({path.stat().st_size / 1024:.0f} KiB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
