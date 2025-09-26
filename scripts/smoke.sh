#!/usr/bin/env bash
set -euo pipefail
URL="${1:-http://127.0.0.1:8000}"

echo "Health:"
curl -s "$URL/healthz" | jq .

echo -e "\nStats:"
curl -s "$URL/index/stats" | jq .

echo -e "\nSearch (cold):"
curl -s "$URL/search" -H "content-type: application/json" \
  -d '{"query":"astronaut stranded on Mars","k":5,"page":1,"per_page":5}' | jq .

echo -e "\nSearch (hot):"
curl -s "$URL/search" -H "content-type: application/json" \
  -d '{"query":"astronaut stranded on Mars","k":5,"page":1,"per_page":5}' | jq .
