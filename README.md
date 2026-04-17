# BIM.AI

Нейросетевая система для генерации BIM-семейств (.rfa) Autodesk Revit по текстовому описанию.

## Что это

Инженер пишет на русском или английском:

> "Приточный потолочный диффузор Ø200мм, расход 150 л/с, Systemair"

→ Система генерирует JSON-спецификацию семейства, которая затем конвертируется в `.rfa` файл с правильными коннекторами, параметрами и типоразмерами.

## Фокус

Инженерные сети (MEP):
- HVAC — вентиляция и кондиционирование
- Piping — водоснабжение, отопление, канализация
- Electrical — электрика и освещение
- Fire Protection — пожаротушение

## Архитектура

```
Звено 1        Звено 2        Звено 3        Звено 4         Звено 5
ДАННЫЕ    →    ОБУЧЕНИЕ   →   ИНФЕРЕНС   →   КОНВЕРТАЦИЯ  →  ИНТЕРФЕЙС
(датасет)      (fine-tune)    (API)          (JSON→.rfa)     (веб/бот)
```

## Модули

| Файл | Назначение |
|---|---|
| `mep_schema.py` | JSON-схема MEP-семейства (17 полей, 23 категории Revit) |
| `mep_generator.py` | Синтетический генератор датасета (15 шаблонов, 4 домена) |
| `mep_prompts.py` | Шаблоны промптов (RU/EN, экспорт Alpaca + ChatML) |
| `mep_validator.py` | Валидатор MEP-ограничений (12 правил) |
| `train_qlora.py` | Обучение через QLoRA + Unsloth |
| `inference.py` | FastAPI-сервер для генерации |
| `eval_pipeline.py` | Автоматическое тестирование модели |

## Быстрый старт

### 1. Установка

```bash
pip install -r requirements.txt  # TODO: создать
# Или вручную:
pip install transformers trl accelerate peft bitsandbytes fastapi uvicorn pydantic
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. Генерация датасета

```bash
python generate_dataset.py --scale 40 --output-dir datasets
# Результат: datasets/mep_chatml.jsonl (~3120 пар)
```

### 3. Обучение (на Vast.ai RTX 4090)

См. [VAST_AI_SETUP.md](VAST_AI_SETUP.md) для полной инструкции.

```bash
python train_qlora.py \
    --data datasets/mep_chatml.jsonl \
    --epochs 4 \
    --batch-size 2 \
    --grad-accum 4 \
    --output ./checkpoints/bim_ai_v1
```

### 4. Оценка качества

```bash
python eval_pipeline.py \
    --adapter ./checkpoints/bim_ai_v1/lora_adapter \
    --n 20 --verbose
```

### 5. Запуск inference API

```bash
python inference.py --adapter ./checkpoints/bim_ai_v1/lora_adapter --port 8000
```

## Техстек

- **Python 3.11+**
- **LLM:** CodeLlama 7B
- **Fine-tuning:** QLoRA через Unsloth (4-bit quantization)
- **GPU:** Vast.ai RTX 4090
- **Inference:** vLLM + FastAPI
- **Конвертация:** Dynamo / Revit API

## Документация

- [BIM_AI_PROJECT.md](BIM_AI_PROJECT.md) — полное описание проекта, архитектура, текущий статус
- [VAST_AI_SETUP.md](VAST_AI_SETUP.md) — инструкция по обучению на Vast.ai

## Статус

🔄 Проект в активной разработке. Готовы: схема данных, генератор, валидатор, скрипты обучения и инференса.
