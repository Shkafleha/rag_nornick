#!/usr/bin/env bash
# Собирает и пушит локально-собираемые образы в registry на ноде.
# Запускать на машине, где есть исходники проекта (с docker).
set -euo pipefail

REGISTRY="${REGISTRY:-192.168.31.241:5000}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

build_push () {
  local name="$1" ; local context="$2" ; local dockerfile="$3"
  echo "==> ${name}"
  docker build -t "${REGISTRY}/llm/${name}:latest" -f "${dockerfile}" "${context}"
  docker push "${REGISTRY}/llm/${name}:latest"
}

build_push rag_api       "${ROOT}/services/rag_api"        "${ROOT}/services/rag_api/Dockerfile"
build_push ui            "${ROOT}/services/ui"             "${ROOT}/services/ui/Dockerfile"
build_push reranker      "${ROOT}/services/reranker"       "${ROOT}/services/reranker/Dockerfile"
build_push indexer_text  "${ROOT}/services/indexer"        "${ROOT}/services/indexer/text_pdf/Dockerfile"
build_push indexer_image "${ROOT}/services/indexer"        "${ROOT}/services/indexer/image_pdf/Dockerfile"

# embeddings: services/embeddings/ отсутствует в репо — пропускаем.
# Если соберёшь — раскомментируй:
# build_push embeddings  "${ROOT}"  "${ROOT}/services/embeddings/Dockerfile"

echo "Готово. Образы в ${REGISTRY}:"
curl -s "http://${REGISTRY}/v2/_catalog"
