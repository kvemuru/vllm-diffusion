#!/usr/bin/env bash
#
# Wait for the vLLM-Omni server, generate one image, and verify it is valid PNG.
#
# Usage:
#   ./scripts/smoke_test.sh [output.png]
#
# Environment:
#   API_BASE    default http://127.0.0.1:8091
#   MODEL_ID    default stabilityai/stable-diffusion-xl-base-1.0
#   TIMEOUT     default 900 seconds
#
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:8091}"
MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
TIMEOUT="${TIMEOUT:-900}"
OUTPUT="${1:-outputs/smoke-test.png}"

mkdir -p "$(dirname "${OUTPUT}")"

echo "Waiting for ${API_BASE}/health (timeout: ${TIMEOUT}s)..."
deadline=$((SECONDS + TIMEOUT))
until curl -fsS "${API_BASE}/health" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    echo "ERROR: server did not become healthy within ${TIMEOUT}s." >&2
    echo "Check logs with: docker compose logs -f sd-server" >&2
    echo "              or: podman logs -f sd-vllm-server" >&2
    exit 1
  fi
  sleep 5
done
echo "Server is healthy."

echo "Generating a test image..."
curl -fsS "${API_BASE}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_ID}\",
    \"prompt\": \"a lighthouse on a rocky coast at dusk, photorealistic, dramatic clouds\",
    \"negative_prompt\": \"blurry, low quality, watermark, text\",
    \"size\": \"1024x1024\",
    \"num_inference_steps\": 30,
    \"guidance_scale\": 7.5,
    \"seed\": 42,
    \"n\": 1,
    \"response_format\": \"b64_json\"
  }" | python3 -c '
import base64, json, sys
result = json.load(sys.stdin)
data = result.get("data") or []
if not data or not data[0].get("b64_json"):
    sys.exit("Unexpected response: " + json.dumps(result)[:1000])
with open(sys.argv[1], "wb") as handle:
    handle.write(base64.b64decode(data[0]["b64_json"]))
' "${OUTPUT}"

python3 - "${OUTPUT}" <<'PY'
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
    sys.exit(f"{path} is not a valid PNG")
width, height = struct.unpack(">II", raw[16:24])
print(f"OK: {path} ({width}x{height}, {len(raw) / 1024:.0f} KiB)")
PY
