#!/usr/bin/env bash
# Раскатывает чарт llm-rag в k3s.
set -euo pipefail

CHART_DIR="${CHART_DIR:-$(cd "$(dirname "$0")/../helm/llm-rag" && pwd)}"
RELEASE="${RELEASE:-llm-rag}"
NS="${NS:-llm}"

if [[ ! -f "${CHART_DIR}/values.secrets.yaml" ]]; then
  echo "ERROR: ${CHART_DIR}/values.secrets.yaml не найден."
  echo "       Скопируй values.secrets.example.yaml и заполни."
  exit 1
fi

helm upgrade --install "${RELEASE}" "${CHART_DIR}" \
  -n "${NS}" --create-namespace \
  -f "${CHART_DIR}/values.yaml" \
  -f "${CHART_DIR}/values.secrets.yaml"

kubectl -n "${NS}" rollout status deployment/rag-api --timeout=5m || true
kubectl -n "${NS}" get pods,svc,ingress
