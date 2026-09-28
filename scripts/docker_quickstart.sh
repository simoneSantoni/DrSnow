#!/usr/bin/env bash
# Build (if needed) and start the DrSnow web GUI in a container.
# Works with Docker (docker compose) or Podman (podman compose / plain podman).
# The GUI is published on http://localhost:8000 (loopback only).
set -euo pipefail

cd "$(dirname "$0")/.."
PORT="${DRSNOW_GUI_PORT:-8000}"
URL="http://localhost:${PORT}/"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    ENGINE=docker
    docker compose up -d --build
elif command -v podman >/dev/null 2>&1; then
    ENGINE=podman
    podman build -t drsnow-gui .
    podman rm -f drsnow-gui >/dev/null 2>&1 || true
    podman run -d --name drsnow-gui --read-only --tmpfs /tmp --cap-drop ALL \
        --security-opt no-new-privileges -p "127.0.0.1:${PORT}:8000" drsnow-gui
else
    echo "Neither Docker (with compose) nor Podman was found." >&2
    echo "Install Docker: https://docs.docker.com/get-docker/" >&2
    exit 1
fi

echo "Waiting for the DrSnow GUI to start (first start compiles code)..."
for _ in $(seq 1 90); do
    if curl -fsS "${URL}healthz" >/dev/null 2>&1; then
        echo "DrSnow GUI is running at ${URL}"
        echo "Stop it with: $([ "$ENGINE" = docker ] && echo 'docker compose down' \
                                                   || echo 'podman rm -f drsnow-gui')"
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$URL" >/dev/null 2>&1 || true
        elif command -v open >/dev/null 2>&1; then
            open "$URL" || true
        fi
        exit 0
    fi
    sleep 2
done
echo "The GUI did not answer on ${URL}healthz; check the container logs:" >&2
echo "  $ENGINE logs drsnow-gui" >&2
exit 1
