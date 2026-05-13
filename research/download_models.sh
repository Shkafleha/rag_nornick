#!/bin/sh
# Скачивает все модели для бенчмарков в hf_cache.
# Запуск:
#   docker compose exec -e HF_TOKEN=<token> notebook-gpu sh /workspace/research/download_models.sh

set -e

EMBEDDERS="
BAAI/bge-m3
deepvk/USER-bge-m3
intfloat/multilingual-e5-large
intfloat/multilingual-e5-base
sergeyzh/LaBSE-ru-turbo
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ai-forever/ru-en-RoSBERTa
cointegrated/LaBSE-en-ru
"

RERANKERS="
BAAI/bge-reranker-v2-m3
BAAI/bge-reranker-base
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
DiTy/cross-encoder-russian-msmarco
amberoad/bert-multilingual-passage-reranking-msmarco
"

# LLM-ы — только HF-варианты (Ollama тянется отдельно через ollama pull)
LLMS="
NuisanceValue/MetalGPT-1-GGUF
"

for m in $EMBEDDERS $RERANKERS; do
    echo "=== $m ==="
    hf download "$m" || echo "!! failed: $m"
done

# MetalGPT: только один GGUF-файл, не весь репо (100 GB)
echo "=== NuisanceValue/MetalGPT-1-GGUF (Q4_K_M only) ==="
hf download NuisanceValue/MetalGPT-1-GGUF MetalGPT-1-32B-Q4_K_M.gguf \
    --local-dir /root/.cache/metalgpt || echo "!! failed: MetalGPT"

echo ""
echo "=== Cache size ==="
du -sh /root/.cache/huggingface/hub /root/.cache/metalgpt 2>/dev/null
