#!/usr/bin/env bash
set -euo pipefail
# Redeploy platex v1.0.4-cpu with host models/configs (run on target Linux host as root).
IMG="${PLATEX_IMAGE:-ghcr.io/vesaaa/platex:v1.0.4-cpu}"
for c in $(docker ps -qf publish=8080); do
  docker stop "$c"
  docker rm "$c"
done
docker run -d --name platex_lpr \
  -p 8080:8080 \
  -v /opt/vscc/platex/configs:/app/configs:ro \
  -v /opt/vscc/platex/models:/app/models \
  -e PLATEX_WORKERS=16 \
  -e PLATEX_URL_MAX_FETCH_CONCURRENCY=32 \
  -e PLATEX_MODEL_POOL_SIZE=8 \
  "$IMG"
sleep 5
curl -fsS "http://127.0.0.1:8080/api/v1/health" | head -c 400
echo
docker ps --format '{{.ID}} {{.Names}} {{.Image}}' | head -3
