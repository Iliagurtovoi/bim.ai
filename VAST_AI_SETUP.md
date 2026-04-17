# BIM.AI — Запуск обучения на Vast.ai

## 1. Аренда инстанса

На vast.ai выбрать:
- GPU: **RTX 4090** (24GB VRAM, достаточно для 7B QLoRA)
- Образ: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel` или `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
- Disk: минимум **50GB** (модель ~14GB + датасет + чекпоинты)
- RAM: минимум 32GB

Ориентировочная стоимость: ~$0.35–0.50/час, обучение ~2–4 часа = ~$1–2 на весь запуск.

## 2. Подготовка окружения

```bash
# Обновить pip и установить зависимости
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl transformers accelerate peft bitsandbytes
pip install fastapi uvicorn pydantic
```

## 3. Загрузка кода и датасета

```bash
# Вариант A: через git (если репозиторий есть)
git clone https://github.com/YOUR_ORG/bim_ai.git
cd bim_ai

# Вариант B: загрузить файлы вручную (scp или веб-интерфейс Vast.ai)
# Нужные файлы:
#   mep_schema.py
#   mep_generator.py
#   mep_validator.py
#   mep_prompts.py
#   train_qlora.py
#   inference.py
#   eval_pipeline.py
#   generate_dataset.py
#   datasets/mep_chatml.jsonl   (или сгенерировать)
```

## 4. Генерация датасета (если не загружен)

```bash
# Генерирует 3120 пар (~24MB) — занимает ~30 секунд
python generate_dataset.py --scale 40 --output-dir datasets
```

## 5. Обучение

### Стандартный запуск (рекомендован):
```bash
python train_qlora.py \
    --data datasets/mep_chatml.jsonl \
    --epochs 4 \
    --batch-size 2 \
    --grad-accum 4 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --output ./checkpoints/bim_ai_v1 \
    --validate
```

### Параметры:
| Параметр | Значение | Описание |
|---|---|---|
| `--epochs` | 4 | Для 3120 пар — достаточно, больше → переобучение |
| `--batch-size` | 2 | RTX 4090 держит 2 без OOM |
| `--grad-accum` | 4 | Эффективный батч = 8 |
| `--lr` | 2e-4 | Стандарт для QLoRA |
| `--lora-r` | 16 | Rank 16 — баланс качество/скорость |
| `--lora-alpha` | 32 | Всегда 2x от lora-r |

### Ожидаемое время:
- 3120 пар × 4 эпохи × ~1 сек/шаг = ~2–3 часа на RTX 4090
- Loss должен упасть с ~2.5 до ~0.3–0.6

### Для более высокого качества:
```bash
python train_qlora.py \
    --lora-r 32 \
    --lora-alpha 64 \
    --epochs 5 \
    --output ./checkpoints/bim_ai_v1_r32
```

## 6. Мониторинг обучения

В логах будет:
```
Step 100/1560 | loss: 1.234 | lr: 1.8e-04
Step 200/1560 | loss: 0.876 | lr: 1.6e-04
...
```

Eval loss на validation set проверяется каждые ~½ эпохи.

## 7. Evaluation после обучения

```bash
python eval_pipeline.py \
    --adapter ./checkpoints/bim_ai_v1/lora_adapter \
    --n 20 \
    --verbose \
    --output ./eval_results.json
```

### Целевые метрики (после 4 эпох на 3120 парах):
| Метрика | Цель |
|---|---|
| JSON validity | ≥ 85% |
| MEP validity | ≥ 70% |
| Domain match | ≥ 90% |
| Schema completeness | ≥ 75% |

## 8. Запуск inference сервера

```bash
python inference.py \
    --adapter ./checkpoints/bim_ai_v1/lora_adapter \
    --port 8000
```

После запуска API доступен:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Потолочный диффузор Ø200мм, расход 150 л/с, Systemair"}'
```

## 9. Сохранение результатов

После обучения скачать с Vast.ai:
```bash
# На локальной машине:
scp -r root@VASTAI_IP:~/bim_ai/checkpoints/bim_ai_v1/lora_adapter ./
scp root@VASTAI_IP:~/bim_ai/eval_results.json ./
```

## 10. Структура checkpoints/

```
checkpoints/bim_ai_v1/
├── lora_adapter/           ← основной артефакт (~400MB)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── training_config.json    ← параметры обучения
├── checkpoint-780/         ← промежуточные чекпоинты
├── checkpoint-1560/
└── checkpoint-2340/
```

## Troubleshooting

**OOM (Out of Memory):**
```bash
# Уменьшить batch size и/или seq length
python train_qlora.py --batch-size 1 --grad-accum 8
```

**Медленная загрузка модели:**
```bash
# Установить HuggingFace кэш на fast SSD
export HF_HOME=/workspace/hf_cache
```

**Модель генерирует не JSON:**
- Убедиться что tokenizer.apply_chat_template работает корректно
- Попробовать другой базовый model: `--model mistralai/Mistral-7B-Instruct-v0.2`

**Loss не падает:**
- Проверить формат датасета: `python -c "import json; d=json.load(open('datasets/mep_chatml.jsonl')); print(d[0])"`
- Увеличить LR: `--lr 3e-4`
- Увеличить LoRA rank: `--lora-r 32`
