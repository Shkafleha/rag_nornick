#!/usr/bin/env bash
# Полный раскат на ноде: bootstrap → build → secrets → deploy.
# Запускать ОДИН РАЗ на 192.168.31.241 после scp проекта.
#
# Usage:
#   cd ~/LLM
#   sudo bash deploy/scripts/00-deploy-all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NODE_IP="${NODE_IP:-192.168.31.241}"
REGISTRY="${REGISTRY:-${NODE_IP}:5000}"
NS="${NS:-llm}"
CHART="${ROOT}/deploy/helm/llm-rag"

echo "================================================="
echo " Шаг 1: bootstrap ноды (k3s + nvidia + registry)"
echo "================================================="
if ! command -v kubectl &>/dev/null; then
  bash "${ROOT}/deploy/scripts/01-bootstrap-node.sh"
else
  echo "(k3s уже установлен, пропускаю)"
fi

echo "================================================="
echo " Шаг 2: helm"
echo "================================================="
if ! command -v helm &>/dev/null; then
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

echo "================================================="
echo " Шаг 3: сборка и пуш образов в localhost registry"
echo "================================================="
REGISTRY="${REGISTRY}" bash "${ROOT}/deploy/scripts/02-build-and-push.sh"

echo "================================================="
echo " Шаг 4: секреты"
echo "================================================="
if [[ ! -f "${CHART}/values.secrets.yaml" ]]; then
  if [[ -f "${ROOT}/.env" ]]; then
    echo "Генерирую values.secrets.yaml из .env"
    LP=$(grep ^LANGFUSE_PUBLIC_KEY "${ROOT}/.env" | cut -d= -f2-)
    LS=$(grep ^LANGFUSE_SECRET_KEY "${ROOT}/.env" | cut -d= -f2-)
    HF=$(grep ^HF_TOKEN          "${ROOT}/.env" | cut -d= -f2-)
    cat > "${CHART}/values.secrets.yaml" <<EOF
secrets:
  langfusePublicKey: "${LP}"
  langfuseSecretKey: "${LS}"
  hfToken: "${HF}"
  clearmlAccessKey: ""
  clearmlSecretKey: ""
EOF
  else
    cp "${CHART}/values.secrets.example.yaml" "${CHART}/values.secrets.yaml"
    echo "WARNING: заполни ${CHART}/values.secrets.yaml и перезапусти helm."
  fi
fi

echo "================================================="
echo " Шаг 5: миграция данных (если есть docker compose)"
echo "================================================="
if docker ps --format '{{.Names}}' | grep -q llm_postgres; then
  bash "${ROOT}/deploy/scripts/03-migrate-data.sh" || true
else
  echo "(docker compose стека нет — пропускаю)"
  # Создаём пустые директории под hostPath
  sudo mkdir -p /var/lib/llm/{hf-cache,paddlex,data,workspace}
  if [[ -d "${ROOT}/data" ]]; then
    sudo rsync -a "${ROOT}/data/" /var/lib/llm/data/
  fi
  sudo chmod -R 777 /var/lib/llm
fi

echo "================================================="
echo " Шаг 6: helm install"
echo "================================================="
helm upgrade --install llm-rag "${CHART}" \
  -n "${NS}" --create-namespace \
  -f "${CHART}/values.yaml" \
  -f "${CHART}/values.secrets.yaml"

echo
echo "================================================="
echo " Готово. Состояние:"
echo "================================================="
kubectl -n "${NS}" get pods,svc,ingress

cat <<EOF

Дальше:
  - kubectl -n ${NS} get pods -w     # дождись Running
  - kubectl -n ${NS} logs -l app.kubernetes.io/name=rag-api -f
  - UI:        http://${NODE_IP}:8501  (или ui.llm.local через ingress)
  - API:       http://${NODE_IP}/      (через ingress на api.llm.local)
  - Langfuse:  http://langfuse.llm.local/
  Прописать в /etc/hosts на клиенте:
    ${NODE_IP}  ui.llm.local api.llm.local langfuse.llm.local nb.llm.local
EOF
