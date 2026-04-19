#!/bin/bash
# ============================================================================
# BIM.AI — Post-Training Automation Script
# ============================================================================
# Автоматизирует шаги после успешного завершения обучения:
#   1. Проверка что адаптер существует
#   2. Автоматическая оценка качества (eval_pipeline)
#   3. Быстрый ручной тест на примере запроса
#   4. (опционально) Загрузка на HuggingFace Hub
#   5. (опционально) Commit изменений в GitHub
#   6. Подготовка архива для удобного скачивания
#
# Использование:
#   ./post_training.sh [имя_чекпоинта]
#
# Примеры:
#   ./post_training.sh                    # использует bim_ai_v1 по умолчанию
#   ./post_training.sh bim_ai_v1_r32      # использует другой чекпоинт
# ============================================================================

set -e  # прекращать выполнение при любой ошибке

# ─── Настройки ──────────────────────────────────────────────────────────────
CHECKPOINT_NAME="${1:-bim_ai_v1}"
CHECKPOINT_DIR="./checkpoints/${CHECKPOINT_NAME}"
ADAPTER_DIR="${CHECKPOINT_DIR}/lora_adapter"
EVAL_OUTPUT="${CHECKPOINT_DIR}/eval_results.json"
EVAL_SAMPLES=20

# Цвета для красивого вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()    { echo -e "${BLUE}▶${NC} $1"; }
ok()     { echo -e "${GREEN}✓${NC} $1"; }
warn()   { echo -e "${YELLOW}⚠${NC} $1"; }
err()    { echo -e "${RED}✗${NC} $1"; }
section(){ echo -e "\n${BLUE}═══ $1 ═══${NC}"; }

# ─── Шаг 1: Проверки ────────────────────────────────────────────────────────
section "1. Проверки"

if [ ! -d "$ADAPTER_DIR" ]; then
    err "Адаптер не найден: $ADAPTER_DIR"
    err "Проверьте что обучение завершилось успешно и путь правильный"
    exit 1
fi
ok "Адаптер найден: $ADAPTER_DIR"

# Проверяем размер адаптера
ADAPTER_SIZE=$(du -sh "$ADAPTER_DIR" | cut -f1)
ok "Размер адаптера: $ADAPTER_SIZE"

# Проверяем наличие ключевых файлов
for f in adapter_config.json adapter_model.safetensors; do
    if [ ! -f "$ADAPTER_DIR/$f" ]; then
        warn "Отсутствует файл: $f"
    else
        ok "Найден: $f"
    fi
done

# ─── Шаг 2: Evaluation ──────────────────────────────────────────────────────
section "2. Автоматическая оценка качества"

if [ -f "$EVAL_OUTPUT" ]; then
    warn "Результаты оценки уже существуют: $EVAL_OUTPUT"
    read -p "Переоценить? [y/N]: " redo
    if [[ "$redo" != "y" && "$redo" != "Y" ]]; then
        log "Пропускаем evaluation"
    else
        rm -f "$EVAL_OUTPUT"
    fi
fi

if [ ! -f "$EVAL_OUTPUT" ]; then
    log "Запуск eval_pipeline.py на $EVAL_SAMPLES примерах..."
    log "Это займёт 5-10 минут"

    python eval_pipeline.py \
        --adapter "$ADAPTER_DIR" \
        --n "$EVAL_SAMPLES" \
        --verbose \
        --output "$EVAL_OUTPUT"

    if [ -f "$EVAL_OUTPUT" ]; then
        ok "Evaluation завершена: $EVAL_OUTPUT"
    else
        err "Evaluation не сгенерировала файл результатов"
        exit 1
    fi
fi

# ─── Шаг 3: Быстрый ручной тест ─────────────────────────────────────────────
section "3. Быстрый ручной тест"

log "Генерация тестового запроса..."

python << 'PYEOF'
import json, sys
try:
    from unsloth import FastLanguageModel
    import torch

    ADAPTER = "./checkpoints/${CHECKPOINT_NAME:-bim_ai_v1}/lora_adapter".replace(
        "${CHECKPOINT_NAME:-bim_ai_v1}", "bim_ai_v1")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    test_prompt = "Потолочный диффузор Ø200мм, расход 150 л/с, Systemair"

    messages = [
        {"role": "system", "content": "You are BIM.AI. Output valid JSON only."},
        {"role": "user", "content": test_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to("cuda")

    outputs = model.generate(inputs, max_new_tokens=512, temperature=0.3, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    print("\n─── Тестовый запрос ───")
    print(f"Prompt: {test_prompt}")
    print("\n─── Ответ модели ───")
    print(response[:800])
    print("..." if len(response) > 800 else "")

    # Проверяем что JSON валидный
    try:
        json.loads(response.strip().split("```")[0] if "```" not in response else
                   response.split("```json")[1].split("```")[0])
        print("\n✓ JSON валиден")
    except Exception as e:
        print(f"\n⚠ JSON не распарсился: {e}")

except Exception as e:
    print(f"Ошибка теста: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ─── Шаг 4: Архивирование для удобного скачивания ──────────────────────────
section "4. Подготовка архива для скачивания"

ARCHIVE_NAME="bim_ai_${CHECKPOINT_NAME}_$(date +%Y%m%d_%H%M).tar.gz"
ARCHIVE_PATH="/tmp/${ARCHIVE_NAME}"

log "Создание архива: $ARCHIVE_PATH"
tar czf "$ARCHIVE_PATH" \
    -C "$(dirname $CHECKPOINT_DIR)" "$(basename $CHECKPOINT_DIR)/lora_adapter" \
    -C "$(dirname $CHECKPOINT_DIR)" "$(basename $CHECKPOINT_DIR)/training_config.json" 2>/dev/null || true

if [ -f "$EVAL_OUTPUT" ]; then
    tar rzf "$ARCHIVE_PATH" -C "$(dirname $EVAL_OUTPUT)" "$(basename $EVAL_OUTPUT)" 2>/dev/null || true
fi

ARCHIVE_SIZE=$(du -sh "$ARCHIVE_PATH" | cut -f1)
ok "Архив создан: $ARCHIVE_PATH ($ARCHIVE_SIZE)"

# ─── Шаг 5: Загрузка на HuggingFace Hub (опционально) ──────────────────────
section "5. Загрузка на HuggingFace Hub"

read -p "Загрузить адаптер на HuggingFace Hub? [y/N]: " upload_hf
if [[ "$upload_hf" == "y" || "$upload_hf" == "Y" ]]; then
    read -p "Ваш HuggingFace username: " hf_user
    read -p "Название репо (напр. bim-ai-v1): " hf_repo

    log "Загрузка на huggingface.co/${hf_user}/${hf_repo}..."
    huggingface-cli upload "${hf_user}/${hf_repo}" "$ADAPTER_DIR" \
        --repo-type model --private || warn "Загрузка не удалась — проверьте логин"

    if [ $? -eq 0 ]; then
        ok "Загружено: https://huggingface.co/${hf_user}/${hf_repo}"
    fi
else
    log "Пропускаем загрузку на HF"
fi

# ─── Шаг 6: Git commit (опционально) ───────────────────────────────────────
section "6. Git commit изменений"

if git status --porcelain 2>/dev/null | grep -q .; then
    log "Обнаружены несохранённые изменения в git:"
    git status --short

    read -p "Закоммитить и запушить? [y/N]: " do_git
    if [[ "$do_git" == "y" || "$do_git" == "Y" ]]; then
        git add -A
        git commit -m "Post-training: add generate_dataset.py and related scripts"
        git push || warn "Push не удался — проверьте креденциалы git"
    else
        log "Пропускаем git"
    fi
else
    ok "Нет несохранённых изменений"
fi

# ─── Финал — инструкции по скачиванию ──────────────────────────────────────
section "✅ Готово! Инструкции по скачиванию"

IP=$(curl -s ifconfig.me 2>/dev/null || echo "ВАШ_IP")
PORT="${VAST_TCP_PORT_22:-ВАШ_PORT}"

cat << EOF

Выполните на локальной машине (Windows PowerShell):

    # Создать папку под результаты
    mkdir C:\\Users\\user\\bim-ai-results -Force
    cd C:\\Users\\user\\bim-ai-results

    # Скачать готовый архив (одним файлом, удобно!)
    scp -P ${PORT} root@${IP}:${ARCHIVE_PATH} .

    # Разархивировать
    tar -xzf ${ARCHIVE_NAME}

Или по отдельности:

    scp -P ${PORT} -r root@${IP}:/workspace/bim.ai/${ADAPTER_DIR} .
    scp -P ${PORT} root@${IP}:/workspace/bim.ai/${EVAL_OUTPUT} .

После успешного скачивания — выключите инстанс на vast.ai,
чтобы не тратить деньги впустую!
EOF

echo ""
ok "Скрипт завершён. Размер архива: $ARCHIVE_SIZE"
