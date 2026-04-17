# BIM.AI

> 🇫🇷 Français · [🇬🇧 English](#-english) · [🇷🇺 Русский](#-русский)

---

## 🇫🇷 Français

Système neuronal pour générer des familles BIM (.rfa) d'Autodesk Revit à partir d'une description textuelle.

### Qu'est-ce que c'est

L'ingénieur écrit en français, anglais ou russe :

> « Diffuseur de plafond soufflant Ø200mm, débit 150 L/s, Systemair »

→ Le système génère une spécification JSON de la famille, qui est ensuite convertie en fichier `.rfa` avec les bons connecteurs, paramètres et dimensions types.

### Domaine d'application

Réseaux techniques du bâtiment (MEP) :
- **HVAC** — ventilation et climatisation
- **Piping** — plomberie, chauffage, évacuation
- **Electrical** — électricité et éclairage
- **Fire Protection** — protection incendie

### Architecture

```
Maillon 1       Maillon 2       Maillon 3       Maillon 4        Maillon 5
DONNÉES    →    ENTRAÎNEMENT →  INFÉRENCE  →    CONVERSION   →   INTERFACE
(dataset)       (fine-tune)     (API)           (JSON→.rfa)      (web/bot)
```

### Modules

| Fichier | Rôle |
|---|---|
| `mep_schema.py` | Schéma JSON de famille MEP (17 champs, 23 catégories Revit) |
| `mep_generator.py` | Générateur synthétique de dataset (15 modèles, 4 domaines) |
| `mep_prompts.py` | Modèles de prompts (RU/EN/FR, export Alpaca + ChatML) |
| `mep_validator.py` | Validateur des contraintes MEP (12 règles) |
| `train_qlora.py` | Entraînement via QLoRA + Unsloth |
| `inference.py` | Serveur FastAPI pour la génération |
| `eval_pipeline.py` | Tests automatiques du modèle |

### Démarrage rapide

#### 1. Installation

```bash
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

#### 2. Génération du dataset

```bash
python generate_dataset.py --scale 40 --output-dir datasets
# Résultat : datasets/mep_chatml.jsonl (~3120 paires)
```

#### 3. Entraînement (sur Vast.ai RTX 4090)

Voir [VAST_AI_SETUP.md](VAST_AI_SETUP.md) pour les instructions complètes.

```bash
python train_qlora.py \
    --data datasets/mep_chatml.jsonl \
    --epochs 4 --batch-size 2 --grad-accum 4 \
    --output ./checkpoints/bim_ai_v1
```

#### 4. Évaluation

```bash
python eval_pipeline.py --adapter ./checkpoints/bim_ai_v1/lora_adapter --n 20 --verbose
```

#### 5. Serveur d'inférence

```bash
python inference.py --adapter ./checkpoints/bim_ai_v1/lora_adapter --port 8000
```

### Stack technique

- **Python 3.11+**
- **LLM :** CodeLlama 7B
- **Fine-tuning :** QLoRA via Unsloth (quantification 4-bit)
- **GPU :** Vast.ai RTX 4090
- **Inférence :** vLLM + FastAPI
- **Conversion :** Dynamo / Revit API

### Documentation

- [BIM_AI_PROJECT.md](BIM_AI_PROJECT.md) — description complète, architecture, état actuel
- [VAST_AI_SETUP.md](VAST_AI_SETUP.md) — guide d'entraînement sur Vast.ai

### Statut

🔄 Projet en développement actif. Déjà prêts : schéma de données, générateur, validateur, scripts d'entraînement et d'inférence.

---

## 🇬🇧 English

Neural network system for generating BIM families (.rfa) for Autodesk Revit from textual descriptions.

### What it does

The engineer writes in English, French, or Russian:

> "Supply ceiling diffuser Ø200mm, flow rate 150 L/s, Systemair"

→ The system generates a JSON specification of the family, which is then converted into a `.rfa` file with proper connectors, parameters, and size types.

### Scope

Building services (MEP):
- **HVAC** — heating, ventilation, air conditioning
- **Piping** — water supply, heating, drainage
- **Electrical** — power and lighting
- **Fire Protection** — sprinklers and alarms

### Architecture

```
Stage 1         Stage 2         Stage 3         Stage 4          Stage 5
DATA      →     TRAINING   →    INFERENCE  →    CONVERSION  →    INTERFACE
(dataset)       (fine-tune)     (API)           (JSON→.rfa)      (web/bot)
```

### Modules

| File | Purpose |
|---|---|
| `mep_schema.py` | JSON schema for MEP families (17 fields, 23 Revit categories) |
| `mep_generator.py` | Synthetic dataset generator (15 templates, 4 domains) |
| `mep_prompts.py` | Prompt templates (RU/EN/FR, Alpaca + ChatML export) |
| `mep_validator.py` | MEP constraint validator (12 rules) |
| `train_qlora.py` | Training via QLoRA + Unsloth |
| `inference.py` | FastAPI server for generation |
| `eval_pipeline.py` | Automated model testing |

### Quick start

#### 1. Install

```bash
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

#### 2. Generate dataset

```bash
python generate_dataset.py --scale 40 --output-dir datasets
# Result: datasets/mep_chatml.jsonl (~3120 pairs)
```

#### 3. Train (on Vast.ai RTX 4090)

See [VAST_AI_SETUP.md](VAST_AI_SETUP.md) for the complete guide.

```bash
python train_qlora.py \
    --data datasets/mep_chatml.jsonl \
    --epochs 4 --batch-size 2 --grad-accum 4 \
    --output ./checkpoints/bim_ai_v1
```

#### 4. Evaluate

```bash
python eval_pipeline.py --adapter ./checkpoints/bim_ai_v1/lora_adapter --n 20 --verbose
```

#### 5. Run inference server

```bash
python inference.py --adapter ./checkpoints/bim_ai_v1/lora_adapter --port 8000
```

### Tech stack

- **Python 3.11+**
- **LLM:** CodeLlama 7B
- **Fine-tuning:** QLoRA via Unsloth (4-bit quantization)
- **GPU:** Vast.ai RTX 4090
- **Inference:** vLLM + FastAPI
- **Conversion:** Dynamo / Revit API

### Documentation

- [BIM_AI_PROJECT.md](BIM_AI_PROJECT.md) — full project description, architecture, current status
- [VAST_AI_SETUP.md](VAST_AI_SETUP.md) — training guide for Vast.ai

### Status

🔄 Actively under development. Ready: data schema, generator, validator, training and inference scripts.

---

## 🇷🇺 Русский

Нейросетевая система для генерации BIM-семейств (.rfa) Autodesk Revit по текстовому описанию.

### Что это

Инженер пишет на русском, английском или французском:

> «Приточный потолочный диффузор Ø200мм, расход 150 л/с, Systemair»

→ Система генерирует JSON-спецификацию семейства, которая затем конвертируется в `.rfa` файл с правильными коннекторами, параметрами и типоразмерами.

### Фокус

Инженерные сети (MEP):
- **HVAC** — вентиляция и кондиционирование
- **Piping** — водоснабжение, отопление, канализация
- **Electrical** — электрика и освещение
- **Fire Protection** — пожаротушение

### Архитектура

```
Звено 1         Звено 2         Звено 3         Звено 4          Звено 5
ДАННЫЕ    →     ОБУЧЕНИЕ   →    ИНФЕРЕНС   →    КОНВЕРТАЦИЯ  →   ИНТЕРФЕЙС
(датасет)       (fine-tune)     (API)           (JSON→.rfa)      (веб/бот)
```

### Модули

| Файл | Назначение |
|---|---|
| `mep_schema.py` | JSON-схема MEP-семейства (17 полей, 23 категории Revit) |
| `mep_generator.py` | Синтетический генератор датасета (15 шаблонов, 4 домена) |
| `mep_prompts.py` | Шаблоны промптов (RU/EN/FR, экспорт Alpaca + ChatML) |
| `mep_validator.py` | Валидатор MEP-ограничений (12 правил) |
| `train_qlora.py` | Обучение через QLoRA + Unsloth |
| `inference.py` | FastAPI-сервер для генерации |
| `eval_pipeline.py` | Автоматическое тестирование модели |

### Быстрый старт

#### 1. Установка

```bash
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

#### 2. Генерация датасета

```bash
python generate_dataset.py --scale 40 --output-dir datasets
# Результат: datasets/mep_chatml.jsonl (~3120 пар)
```

#### 3. Обучение (на Vast.ai RTX 4090)

См. [VAST_AI_SETUP.md](VAST_AI_SETUP.md) для полной инструкции.

```bash
python train_qlora.py \
    --data datasets/mep_chatml.jsonl \
    --epochs 4 --batch-size 2 --grad-accum 4 \
    --output ./checkpoints/bim_ai_v1
```

#### 4. Оценка качества

```bash
python eval_pipeline.py --adapter ./checkpoints/bim_ai_v1/lora_adapter --n 20 --verbose
```

#### 5. Запуск inference сервера

```bash
python inference.py --adapter ./checkpoints/bim_ai_v1/lora_adapter --port 8000
```

### Техстек

- **Python 3.11+**
- **LLM:** CodeLlama 7B
- **Fine-tuning:** QLoRA через Unsloth (4-bit quantization)
- **GPU:** Vast.ai RTX 4090
- **Inference:** vLLM + FastAPI
- **Конвертация:** Dynamo / Revit API

### Документация

- [BIM_AI_PROJECT.md](BIM_AI_PROJECT.md) — полное описание проекта, архитектура, текущий статус
- [VAST_AI_SETUP.md](VAST_AI_SETUP.md) — инструкция по обучению на Vast.ai

### Статус

🔄 Проект в активной разработке. Готовы: схема данных, генератор, валидатор, скрипты обучения и инференса.
