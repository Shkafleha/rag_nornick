#!/usr/bin/env bash
# Перенос данных из существующего docker-compose стека в k8s PVC/hostPath.
# Запускать на ноде, где есть и docker, и развёрнутый k8s.
set -euo pipefail

NS="${NS:-llm}"
HOST_DIR="${HOST_DIR:-/var/lib/llm}"

echo "==> Postgres dump → restore"
docker exec llm_postgres pg_dumpall -U postgres > /tmp/pg_dump.sql
POD=$(kubectl -n "${NS}" get pod -l app.kubernetes.io/name=postgres -o jsonpath='{.items[0].metadata.name}')
kubectl -n "${NS}" cp /tmp/pg_dump.sql "${POD}:/tmp/pg_dump.sql"
kubectl -n "${NS}" exec "${POD}" -- psql -U postgres -f /tmp/pg_dump.sql

echo "==> Qdrant snapshot → restore"
# Снимок текущего qdrant:
curl -X POST "http://localhost:6333/collections/_all/snapshots" || true
docker cp qdrant:/qdrant/storage/. /tmp/qdrant-storage/
QPOD=$(kubectl -n "${NS}" get pod -l app.kubernetes.io/name=qdrant -o jsonpath='{.items[0].metadata.name}')
kubectl -n "${NS}" exec "${QPOD}" -- sh -c 'rm -rf /qdrant/storage/*'
kubectl -n "${NS}" cp /tmp/qdrant-storage "${QPOD}:/qdrant/storage"
kubectl -n "${NS}" delete pod "${QPOD}"   # перезапустить на восстановленных данных

echo "==> ClickHouse (если есть данные в langfuse)"
docker exec langfuse_clickhouse clickhouse-client --user clickhouse --password clickhouse \
  --query "BACKUP DATABASE default TO File('/tmp/ch_backup')" || true
docker cp langfuse_clickhouse:/tmp/ch_backup /tmp/ch_backup
CPOD=$(kubectl -n "${NS}" get pod -l app.kubernetes.io/name=clickhouse -o jsonpath='{.items[0].metadata.name}')
kubectl -n "${NS}" cp /tmp/ch_backup "${CPOD}:/tmp/ch_backup"
kubectl -n "${NS}" exec "${CPOD}" -- clickhouse-client --user clickhouse --password clickhouse \
  --query "RESTORE DATABASE default FROM File('/tmp/ch_backup')" || true

echo "==> ./data/ → ${HOST_DIR}/data (для indexer Jobs)"
sudo rsync -a --info=progress2 ./data/ "${HOST_DIR}/data/"

echo "==> hf_cache, paddlex (опционально)"
docker run --rm -v hf_cache:/src -v "${HOST_DIR}/hf-cache":/dst alpine \
  sh -c 'cp -a /src/. /dst/' || true
docker run --rm -v paddlex_cache:/src -v "${HOST_DIR}/paddlex":/dst alpine \
  sh -c 'cp -a /src/. /dst/' || true

echo "Готово. Проверь kubectl -n ${NS} get pods и логи."
