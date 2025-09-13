#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"

echo "• /healthz"
curl -sf "$BASE/healthz" | jq .

echo "• /index/stats"
curl -sf "$BASE/index/stats" | jq .

echo "• /search"
curl -sf "$BASE/search" \
  -H 'content-type: application/json' \
  -d '{"query":"astronaut stranded on Mars","k":5,"page":1,"per_page":5}' | jq .
