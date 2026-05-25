# Миграция моделей Ollama из старого контейнера в named volume
# Запускать ОДИН РАЗ перед docker-compose up

Write-Host "=== Миграция Ollama моделей в named volume ===" -ForegroundColor Cyan

# 1. Проверка что старый контейнер ollama_local запущен
$container = docker ps -a --filter "name=ollama_local" --format "{{.Names}}"
if (-not $container) {
    Write-Host "[INFO] Контейнер ollama_local не найден. Миграция не требуется." -ForegroundColor Yellow
    Write-Host "[INFO] Просто запустите: docker-compose up -d" -ForegroundColor Yellow
    exit 0
}

Write-Host "[1/5] Найден контейнер ollama_local" -ForegroundColor Green

# 2. Создаём named volume (если ещё не создан)
docker volume create llm_ollama_data | Out-Null
Write-Host "[2/5] Создан volume: llm_ollama_data" -ForegroundColor Green

# 3. Копируем данные из контейнера ollama_local в volume
Write-Host "[3/5] Копирую модели (это может занять минуту)..." -ForegroundColor Yellow

# Запускаем временный alpine контейнер с примонтированным volume,
# затем копируем с помощью docker cp напрямую
$tempContainer = docker create --name ollama_migrate_temp -v llm_ollama_data:/dest alpine:latest
docker cp ollama_local:/root/.ollama/. ollama_migrate_temp:/dest/
docker rm ollama_migrate_temp | Out-Null

Write-Host "[3/5] Модели скопированы в volume" -ForegroundColor Green

# 4. Останавливаем и удаляем старый контейнер
Write-Host "[4/5] Останавливаю старый контейнер ollama_local..." -ForegroundColor Yellow
docker stop ollama_local | Out-Null
docker rm ollama_local | Out-Null
Write-Host "[4/5] Старый контейнер удалён" -ForegroundColor Green

# 5. Готово
Write-Host ""
Write-Host "[5/5] Миграция завершена!" -ForegroundColor Green
Write-Host ""
Write-Host "Теперь запустите:" -ForegroundColor Cyan
Write-Host "  docker-compose up -d" -ForegroundColor White
Write-Host ""
Write-Host "После запуска проверьте что модели на месте:" -ForegroundColor Cyan
Write-Host "  docker exec ollama_local ollama list" -ForegroundColor White
