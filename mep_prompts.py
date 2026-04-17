"""
BIM.AI — MEP Prompt Templates for Dataset Builder
====================================================
Генерирует пары (prompt, completion) для fine-tuning LLM на задаче:
    текстовое описание MEP-элемента → JSON-спецификация семейства

Каждый шаблон создаёт вариации:
- Разный уровень детализации (краткий / средний / подробный)
- Русский и английский
- Инженерный жаргон vs формальное описание
- С указанием производителя и без
"""

from __future__ import annotations
import random
import json
from dataclasses import dataclass, field

from mep_schema import MEPDomain, SystemType, ConnectorShape


# ─────────────────────────────────────────────
# Prompt Difficulty Levels
# ─────────────────────────────────────────────

class PromptLevel:
    BRIEF = "brief"           # 1-2 предложения
    STANDARD = "standard"     # 3-5 предложений
    DETAILED = "detailed"     # Полная спецификация с параметрами


# ─────────────────────────────────────────────
# Language templates
# ─────────────────────────────────────────────

@dataclass
class PromptVariant:
    """A single prompt template with slots for substitution."""
    template_ru: str
    template_en: str
    required_fields: list[str] = field(default_factory=list)
    level: str = PromptLevel.STANDARD


# ─────────────────────────────────────────────
# HVAC Prompts
# ─────────────────────────────────────────────

HVAC_DIFFUSER_PROMPTS = [
    # BRIEF
    PromptVariant(
        template_ru="Создай потолочный диффузор {shape}, подключение Ø{duct_diam}мм",
        template_en="Create a ceiling {shape} diffuser, {duct_diam}mm duct connection",
        required_fields=["shape", "duct_diam"],
        level=PromptLevel.BRIEF,
    ),
    PromptVariant(
        template_ru="Нужен приточный {shape} диффузор для системы вентиляции, расход {airflow} л/с",
        template_en="Need a supply {shape} diffuser for HVAC system, {airflow} L/s airflow",
        required_fields=["shape", "airflow"],
        level=PromptLevel.BRIEF,
    ),
    # STANDARD
    PromptVariant(
        template_ru=(
            "Создай семейство Revit: {shape} приточный диффузор потолочного монтажа. "
            "Диаметр подключения к воздуховоду {duct_diam} мм, расход воздуха {airflow} л/с. "
            "Нужны параметры потерь давления и уровня шума. "
            "Дальнобойность струи не менее {throw} м."
        ),
        template_en=(
            "Create a Revit family: {shape} ceiling-mounted supply diffuser. "
            "Duct connection diameter {duct_diam}mm, airflow {airflow} L/s. "
            "Include pressure drop and noise level parameters. "
            "Minimum throw distance {throw}m."
        ),
        required_fields=["shape", "duct_diam", "airflow", "throw"],
        level=PromptLevel.STANDARD,
    ),
    # DETAILED
    PromptVariant(
        template_ru=(
            "Создай параметрическое семейство Revit для потолочного {shape} приточного диффузора:\n"
            "- Производитель: {manufacturer}\n"
            "- Подключение к воздуховоду: {shape_detail}, Ø{duct_diam} мм\n"
            "- Расход воздуха: {airflow} л/с, потери давления: {pressure_drop} Па\n"
            "- Уровень шума: NC {noise_level}\n"
            "- Регулируемые лопатки\n"
            "- Типоразмерный ряд: Ø{sizes}\n"
            "- Классификация: OmniClass {omniclass}, IFC {ifc_class}\n"
            "- Материал: {material}"
        ),
        template_en=(
            "Create a parametric Revit family for ceiling-mounted {shape} supply diffuser:\n"
            "- Manufacturer: {manufacturer}\n"
            "- Duct connection: {shape_detail}, Ø{duct_diam}mm\n"
            "- Airflow: {airflow} L/s, pressure drop: {pressure_drop} Pa\n"
            "- Noise level: NC {noise_level}\n"
            "- Adjustable blades\n"
            "- Size range: Ø{sizes}\n"
            "- Classification: OmniClass {omniclass}, IFC {ifc_class}\n"
            "- Material: {material}"
        ),
        required_fields=["shape", "shape_detail", "duct_diam", "airflow",
                         "pressure_drop", "noise_level", "sizes", "manufacturer",
                         "omniclass", "ifc_class", "material"],
        level=PromptLevel.DETAILED,
    ),
]

HVAC_FCU_PROMPTS = [
    PromptVariant(
        template_ru="Создай фанкойл {config} на {cooling}кВт холода, потолочный монтаж",
        template_en="Create {config} fan coil unit, {cooling}kW cooling, ceiling mounted",
        required_fields=["config", "cooling"],
        level=PromptLevel.BRIEF,
    ),
    PromptVariant(
        template_ru=(
            "Семейство Revit — канальный фанкойл ({config}):\n"
            "- Холодопроизводительность: {cooling} кВт\n"
            "- Теплопроизводительность: {heating} кВт\n"
            "- Расход воздуха: {airflow} л/с\n"
            "- Подключения: приточный и обратный воздуховоды Ø{duct_diam} мм, "
            "трубы ХВС/ГВС {pipe_dn}\n"
            "- Электропитание: {voltage}В / {power}Вт"
        ),
        template_en=(
            "Revit family — ducted fan coil unit ({config}):\n"
            "- Cooling capacity: {cooling} kW\n"
            "- Heating capacity: {heating} kW\n"
            "- Airflow: {airflow} L/s\n"
            "- Connections: supply and return ducts Ø{duct_diam}mm, "
            "CHW/HW pipes {pipe_dn}\n"
            "- Electrical: {voltage}V / {power}W"
        ),
        required_fields=["config", "cooling", "heating", "airflow",
                         "duct_diam", "pipe_dn", "voltage", "power"],
        level=PromptLevel.STANDARD,
    ),
]

HVAC_VAV_PROMPTS = [
    PromptVariant(
        template_ru="VAV-бокс круглый Ø{diam}мм, макс расход {max_flow} л/с, {reheat}",
        template_en="Round VAV box Ø{diam}mm, max airflow {max_flow} L/s, {reheat}",
        required_fields=["diam", "max_flow", "reheat"],
        level=PromptLevel.BRIEF,
    ),
    PromptVariant(
        template_ru=(
            "Создай семейство VAV-бокса для Revit:\n"
            "- Тип: с регулируемым расходом{reheat_detail}\n"
            "- Диаметр подключения: Ø{diam} мм\n"
            "- Расход: мин {min_flow} — макс {max_flow} л/с\n"
            "- Потери давления: {pdrop} Па\n"
            "- Управление: {control}"
        ),
        template_en=(
            "Create Revit VAV box family:\n"
            "- Type: variable air volume{reheat_detail}\n"
            "- Connection diameter: Ø{diam}mm\n"
            "- Airflow: min {min_flow} — max {max_flow} L/s\n"
            "- Pressure drop: {pdrop} Pa\n"
            "- Control: {control}"
        ),
        required_fields=["diam", "min_flow", "max_flow", "pdrop",
                         "reheat_detail", "control"],
        level=PromptLevel.STANDARD,
    ),
]

HVAC_AHU_PROMPTS = [
    PromptVariant(
        template_ru="Центральный кондиционер (AHU) на {airflow} м³/ч, с рекуперацией",
        template_en="Air handling unit (AHU) {airflow} m³/h with heat recovery",
        required_fields=["airflow"],
        level=PromptLevel.BRIEF,
    ),
    PromptVariant(
        template_ru=(
            "Семейство Revit — приточно-вытяжная установка:\n"
            "- Расход: {airflow} м³/ч\n"
            "- Напор вентилятора: {esp} Па\n"
            "- Холодоснабжение: {cooling}кВт, теплоснабжение: {heating}кВт\n"
            "- Рекуперация: {recovery}%, тип {recovery_type}\n"
            "- Фильтрация: {filter_class}\n"
            "- Коннекторы: приток, вытяжка, наружный, удаляемый (прямоугольные), "
            "подключения CHW и HW (трубные)\n"
            "- Производитель: {manufacturer}"
        ),
        template_en=(
            "Revit family — air handling unit:\n"
            "- Airflow: {airflow} m³/h\n"
            "- Fan ESP: {esp} Pa\n"
            "- Cooling: {cooling}kW, heating: {heating}kW\n"
            "- Heat recovery: {recovery}%, type {recovery_type}\n"
            "- Filtration: {filter_class}\n"
            "- Connectors: supply, return, outside, exhaust (rectangular ducts), "
            "CHW and HW pipe connections\n"
            "- Manufacturer: {manufacturer}"
        ),
        required_fields=["airflow", "esp", "cooling", "heating", "recovery",
                         "recovery_type", "filter_class", "manufacturer"],
        level=PromptLevel.DETAILED,
    ),
]


# ─────────────────────────────────────────────
# Piping Prompts
# ─────────────────────────────────────────────

PIPING_FIXTURE_PROMPTS = [
    PromptVariant(
        template_ru="Умывальник настенный {width}мм, подключения ХВС/ГВС и канализация",
        template_en="Wall-mounted washbasin {width}mm, CW/HW and drain connections",
        required_fields=["width"],
        level=PromptLevel.BRIEF,
    ),
    PromptVariant(
        template_ru=(
            "Семейство Revit — умывальник {type}:\n"
            "- Размеры: {width}×{depth} мм, высота установки {height} мм\n"
            "- Подключения: ХВС DN{cw_dn}, ГВС DN{hw_dn}, канализация DN{drain_dn}\n"
            "- Санитарных приборов: {fu} FU\n"
            "- Расход воды: {flow} л/с\n"
            "- Монтаж: {host}"
        ),
        template_en=(
            "Revit family — {type} washbasin:\n"
            "- Dimensions: {width}×{depth}mm, mounting height {height}mm\n"
            "- Connections: CW DN{cw_dn}, HW DN{hw_dn}, drain DN{drain_dn}\n"
            "- Fixture units: {fu} FU\n"
            "- Flow rate: {flow} L/s\n"
            "- Hosting: {host}"
        ),
        required_fields=["type", "width", "depth", "height",
                         "cw_dn", "hw_dn", "drain_dn", "fu", "flow", "host"],
        level=PromptLevel.STANDARD,
    ),
]

PIPING_PUMP_PROMPTS = [
    PromptVariant(
        template_ru="Циркуляционный насос {flow} м³/ч, напор {head} м",
        template_en="Circulation pump {flow} m³/h, head {head}m",
        required_fields=["flow", "head"],
        level=PromptLevel.BRIEF,
    ),
    PromptVariant(
        template_ru=(
            "Семейство Revit — насос циркуляционный:\n"
            "- Расход: {flow} м³/ч, напор: {head} м\n"
            "- Мощность двигателя: {power} кВт\n"
            "- Подключения: вход DN{inlet_dn}, выход DN{outlet_dn}\n"
            "- Тип: {pump_type}\n"
            "- Производитель: {manufacturer}"
        ),
        template_en=(
            "Revit family — circulation pump:\n"
            "- Flow: {flow} m³/h, head: {head}m\n"
            "- Motor power: {power}kW\n"
            "- Connections: inlet DN{inlet_dn}, outlet DN{outlet_dn}\n"
            "- Type: {pump_type}\n"
            "- Manufacturer: {manufacturer}"
        ),
        required_fields=["flow", "head", "power", "inlet_dn", "outlet_dn",
                         "pump_type", "manufacturer"],
        level=PromptLevel.STANDARD,
    ),
]


# ─────────────────────────────────────────────
# Electrical Prompts
# ─────────────────────────────────────────────

ELECTRICAL_PANEL_PROMPTS = [
    PromptVariant(
        template_ru="Распределительный щит {amps}A, {ways} модулей",
        template_en="Distribution panel {amps}A, {ways} ways",
        required_fields=["amps", "ways"],
        level=PromptLevel.BRIEF,
    ),
    PromptVariant(
        template_ru=(
            "Семейство Revit — электрический щит:\n"
            "- Номинал: {amps}А, {voltage}В, {phases}-фазный\n"
            "- Количество модулей: {ways}\n"
            "- Ток КЗ: {sc_rating} кА\n"
            "- Подключение: кабель {cable_size}мм², {conduit_size}мм\n"
            "- Степень защиты: IP{ip_rating}\n"
            "- Монтаж: {host}"
        ),
        template_en=(
            "Revit family — electrical distribution panel:\n"
            "- Rating: {amps}A, {voltage}V, {phases}-phase\n"
            "- Number of ways: {ways}\n"
            "- Short circuit rating: {sc_rating}kA\n"
            "- Connection: cable {cable_size}mm², {conduit_size}mm conduit\n"
            "- IP rating: IP{ip_rating}\n"
            "- Hosting: {host}"
        ),
        required_fields=["amps", "voltage", "phases", "ways", "sc_rating",
                         "cable_size", "conduit_size", "ip_rating", "host"],
        level=PromptLevel.DETAILED,
    ),
]

ELECTRICAL_LIGHT_PROMPTS = [
    PromptVariant(
        template_ru="Светильник LED потолочный {wattage}Вт, {lumens} лм",
        template_en="Ceiling LED fixture {wattage}W, {lumens}lm",
        required_fields=["wattage", "lumens"],
        level=PromptLevel.BRIEF,
    ),
    PromptVariant(
        template_ru=(
            "Семейство Revit — светодиодный светильник:\n"
            "- Мощность: {wattage}Вт, световой поток: {lumens} лм\n"
            "- Цветовая температура: {cct}K, CRI ≥ {cri}\n"
            "- Тип: {fixture_type}, монтаж: {host}\n"
            "- Диммирование: {dimming}\n"
            "- Питание: {voltage}В, подключение через кондуит Ø{conduit}мм"
        ),
        template_en=(
            "Revit family — LED light fixture:\n"
            "- Power: {wattage}W, luminous flux: {lumens}lm\n"
            "- Color temperature: {cct}K, CRI ≥ {cri}\n"
            "- Type: {fixture_type}, mounting: {host}\n"
            "- Dimming: {dimming}\n"
            "- Power supply: {voltage}V, {conduit}mm conduit connection"
        ),
        required_fields=["wattage", "lumens", "cct", "cri",
                         "fixture_type", "host", "dimming", "voltage", "conduit"],
        level=PromptLevel.STANDARD,
    ),
]


# ─────────────────────────────────────────────
# Fire Protection Prompts
# ─────────────────────────────────────────────

FIRE_SPRINKLER_PROMPTS = [
    PromptVariant(
        template_ru="Спринклер пендентный K{k_factor}, {temp}°C",
        template_en="Pendant sprinkler K{k_factor}, {temp}°C activation",
        required_fields=["k_factor", "temp"],
        level=PromptLevel.BRIEF,
    ),
    PromptVariant(
        template_ru=(
            "Семейство Revit — спринклерная головка:\n"
            "- Тип: {sprinkler_type}, монтаж: {orientation}\n"
            "- K-фактор: {k_factor}, температура активации: {temp}°C\n"
            "- Зона покрытия: {coverage} м²\n"
            "- Подключение: резьба DN{thread_dn}\n"
            "- Система: {system_type}\n"
            "- Производитель: {manufacturer}"
        ),
        template_en=(
            "Revit family — sprinkler head:\n"
            "- Type: {sprinkler_type}, mounting: {orientation}\n"
            "- K-factor: {k_factor}, activation temperature: {temp}°C\n"
            "- Coverage area: {coverage}m²\n"
            "- Connection: DN{thread_dn} threaded\n"
            "- System: {system_type}\n"
            "- Manufacturer: {manufacturer}"
        ),
        required_fields=["sprinkler_type", "orientation", "k_factor", "temp",
                         "coverage", "thread_dn", "system_type", "manufacturer"],
        level=PromptLevel.STANDARD,
    ),
]

FIRE_DETECTOR_PROMPTS = [
    PromptVariant(
        template_ru="Дымовой извещатель {detector_type}, покрытие {coverage} м²",
        template_en="{detector_type} smoke detector, {coverage}m² coverage",
        required_fields=["detector_type", "coverage"],
        level=PromptLevel.BRIEF,
    ),
    PromptVariant(
        template_ru=(
            "Семейство Revit — пожарный извещатель:\n"
            "- Тип: {detector_type}\n"
            "- Зона покрытия: {coverage} м², макс высота потолка: {max_height} м\n"
            "- Питание: {voltage}В DC, шлейф {cable}мм²\n"
            "- Протокол: {protocol}\n"
            "- Монтаж: потолочный"
        ),
        template_en=(
            "Revit family — fire detector:\n"
            "- Type: {detector_type}\n"
            "- Coverage: {coverage}m², max ceiling height: {max_height}m\n"
            "- Power: {voltage}V DC, loop cable {cable}mm²\n"
            "- Protocol: {protocol}\n"
            "- Mounting: ceiling"
        ),
        required_fields=["detector_type", "coverage", "max_height",
                         "voltage", "cable", "protocol"],
        level=PromptLevel.STANDARD,
    ),
]


# ─────────────────────────────────────────────
# Prompt Value Pools (for slot filling)
# ─────────────────────────────────────────────

VALUE_POOLS = {
    "shape": ["круглый|round", "квадратный|square"],
    "shape_detail": ["круглое сечение|round section", "прямоугольное сечение|rectangular section"],
    "duct_diam": [100, 125, 150, 200, 250, 300, 315, 400],
    "airflow": [50, 80, 100, 150, 200, 250, 300, 400, 500],
    "throw": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    "pressure_drop": [15, 20, 25, 30, 40, 50, 60],
    "noise_level": [18, 20, 25, 28, 30, 35],
    "sizes": ["100-200-300-400", "150-250-315-400", "200-300-400-500"],
    "manufacturer": ["Systemair", "Trox", "Swegon", "Lindab", "Daikin", "Carrier",
                      "Grundfos", "Wilo", "ABB", "Schneider", "Viking", "Hochiki"],
    "omniclass": ["23-33 21 11 11", "23-33 21 11 15", "23-33 19 11"],
    "ifc_class": ["IfcAirTerminal", "IfcUnitaryEquipment", "IfcPump"],
    "material": ["Алюминий|Aluminum", "Оцинкованная сталь|Galvanized steel",
                  "Нержавеющая сталь|Stainless steel", "Пластик|Plastic"],
    "config": ["2-трубный|2-pipe", "4-трубный|4-pipe"],
    "cooling": [2, 3.5, 5, 7, 10, 15],
    "heating": [3, 5, 7, 10, 15, 20],
    "pipe_dn": ["DN20", "DN25", "DN32", "DN40"],
    "voltage": [220, 230, 380, 400],
    "power": [30, 50, 80, 120, 180, 250],
    "diam": [150, 200, 250, 300, 400],
    "max_flow": [100, 200, 300, 500, 800, 1200],
    "min_flow": [30, 60, 90, 150, 240],
    "pdrop": [25, 35, 50, 70, 100],
    "reheat": ["без догрева|no reheat", "с электродогревом|with electric reheat",
               "с водяным догревом|with hot water reheat"],
    "reheat_detail": ["|", " с электродогревом| with electric reheat",
                      " с водяным догревом| with hot water reheat"],
    "control": ["DDC", "0-10V", "BACnet", "Modbus"],
    "esp": [300, 500, 800, 1000, 1200],
    "recovery": [70, 75, 80, 85],
    "recovery_type": ["роторный|rotary", "пластинчатый|plate", "рекуперативный|recuperative"],
    "filter_class": ["F7", "F9", "H13", "H14"],
    "width": [400, 500, 600, 700],
    "depth": [300, 400, 450, 500],
    "height": [750, 800, 850, 900],
    "cw_dn": [15, 20], "hw_dn": [15, 20], "drain_dn": [32, 40, 50],
    "fu": [0.5, 1, 1.5, 2],
    "flow": [0.1, 0.15, 0.2, 0.3],
    "head": [3, 5, 8, 12, 20],
    "inlet_dn": [25, 32, 40, 50, 65, 80],
    "outlet_dn": [25, 32, 40, 50, 65, 80],
    "pump_type": ["мокрый ротор|wet rotor", "сухой ротор|dry rotor", "inline|inline"],
    "type": ["стандартный|standard", "компактный|compact", "для МГН|accessible"],
    "host": ["стена|wall", "потолок|ceiling", "пол|floor"],
    "amps": [63, 100, 160, 250, 400, 630],
    "ways": [12, 24, 36, 48, 72],
    "phases": [1, 3],
    "sc_rating": [10, 25, 36, 50, 65],
    "cable_size": [10, 16, 25, 35, 50, 70, 95],
    "conduit_size": [20, 25, 32, 40, 50, 63],
    "conduit": [16, 20, 25],
    "ip_rating": ["31", "41", "54", "55", "65"],
    "wattage": [18, 36, 45, 60, 72, 120],
    "lumens": [2000, 3600, 4200, 5400, 7200, 9000, 14000],
    "cct": [2700, 3000, 4000, 5000, 6500],
    "cri": [80, 90, 95],
    "fixture_type": ["панельный|panel", "даунлайт|downlight", "линейный|linear",
                     "трековый|track", "прожектор|floodlight"],
    "dimming": ["DALI", "0-10V", "нет|none", "Zigbee"],
    "k_factor": [57, 80, 115, 161, 202, 242, 363],
    "temp": [57, 68, 79, 93, 141],
    "coverage": [9, 12, 16, 21],
    "thread_dn": [15, 20],
    "sprinkler_type": ["стандартный|standard response", "быстродействующий|quick response", "ESFR|ESFR"],
    "orientation": ["пендентный|pendant", "розеттный|upright", "скрытый|concealed"],
    "system_type": ["водозаполненная|wet pipe", "сухотрубная|dry pipe", "предварительного действия|pre-action"],
    "detector_type": ["оптический|optical", "тепловой|thermal", "мультисенсорный|multi-sensor",
                      "аспирационный|aspirating"],
    "max_height": [4, 6, 8, 10, 12],
    "cable": [1.0, 1.5, 2.5],
    "protocol": ["адресный|addressable", "пороговый|conventional", "аналоговый|analog"],
}


# ─────────────────────────────────────────────
# All Template Groups
# ─────────────────────────────────────────────

ALL_PROMPT_GROUPS = {
    "hvac_diffuser": HVAC_DIFFUSER_PROMPTS,
    "hvac_fcu": HVAC_FCU_PROMPTS,
    "hvac_vav": HVAC_VAV_PROMPTS,
    "hvac_ahu": HVAC_AHU_PROMPTS,
    "piping_fixture": PIPING_FIXTURE_PROMPTS,
    "piping_pump": PIPING_PUMP_PROMPTS,
    "electrical_panel": ELECTRICAL_PANEL_PROMPTS,
    "electrical_light": ELECTRICAL_LIGHT_PROMPTS,
    "fire_sprinkler": FIRE_SPRINKLER_PROMPTS,
    "fire_detector": FIRE_DETECTOR_PROMPTS,
}


# ─────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────

class MEPPromptBuilder:
    """
    Builds (prompt, completion) pairs for LLM fine-tuning.
    Combines prompt templates with generated MEP families.
    """

    def __init__(self, lang: str = "ru", seed: int | None = None):
        self.lang = lang
        if seed is not None:
            random.seed(seed)

    def _pick_value(self, field_name: str) -> str | int | float:
        """Pick a random value for a prompt slot."""
        pool = VALUE_POOLS.get(field_name)
        if pool is None:
            return f"<{field_name}>"

        val = random.choice(pool)
        if isinstance(val, str) and "|" in val:
            parts = val.split("|")
            return parts[0] if self.lang == "ru" else parts[1]
        return val

    def fill_template(self, variant: PromptVariant) -> str:
        """Fill a prompt template with random values."""
        template = variant.template_ru if self.lang == "ru" else variant.template_en
        values = {}
        for f in variant.required_fields:
            values[f] = self._pick_value(f)
        try:
            return template.format(**values)
        except KeyError as e:
            return template  # Fallback if a field is missing

    def build_training_pair(
        self,
        prompt_variant: PromptVariant,
        family_json: dict,
    ) -> dict:
        """
        Build a single training pair:
          - instruction: filled prompt template
          - output: JSON family spec
        """
        instruction = self.fill_template(prompt_variant)

        # System prompt for the LLM
        system = (
            "You are BIM.AI, an expert assistant for generating Revit MEP family specifications. "
            "Given a description of an MEP component, output a complete JSON specification "
            "following the MEPBIMFamily schema. Include connectors with proper flow directions, "
            "system types, and dimensions. Include calculation parameters with formulas where applicable."
        )

        return {
            "system": system,
            "instruction": instruction,
            "output": json.dumps(family_json, indent=2, ensure_ascii=False),
            "metadata": {
                "domain": family_json.get("mep_domain", ""),
                "category": family_json.get("category", ""),
                "level": prompt_variant.level,
                "lang": self.lang,
            },
        }

    def build_dataset(
        self,
        families: list[dict],
        prompts_per_family: int = 3,
    ) -> list[dict]:
        """
        Build a complete training dataset from generated families.

        For each family, picks matching prompt templates and generates
        multiple (prompt, completion) pairs at different detail levels.
        """
        dataset = []

        # Map families to prompt groups by domain+category heuristic
        domain_category_map = {
            ("HVAC", "Air Terminals"): ["hvac_diffuser"],
            ("HVAC", "Mechanical Equipment"): ["hvac_fcu", "hvac_ahu"],
            ("HVAC", "Duct Accessories"): ["hvac_vav"],
            ("Piping", "Plumbing Fixtures"): ["piping_fixture"],
            ("Piping", "Plumbing Equipment"): ["piping_pump"],
            ("Electrical", "Electrical Equipment"): ["electrical_panel"],
            ("Electrical", "Lighting Fixtures"): ["electrical_light"],
            ("FireProtection", "Sprinklers"): ["fire_sprinkler"],
            ("FireProtection", "Fire Alarm Devices"): ["fire_detector"],
        }

        for family in families:
            domain = family.get("mep_domain", "")
            category = family.get("category", "")
            groups = domain_category_map.get((domain, category), [])

            if not groups:
                # Fallback: use any HVAC or generic template
                groups = ["hvac_diffuser"]

            for group_name in groups:
                prompts = ALL_PROMPT_GROUPS.get(group_name, [])
                selected = random.sample(prompts, min(prompts_per_family, len(prompts)))

                for variant in selected:
                    pair = self.build_training_pair(variant, family)
                    dataset.append(pair)

        return dataset

    def export_for_unsloth(self, dataset: list[dict]) -> list[dict]:
        """
        Convert to Unsloth/Alpaca format for QLoRA training.

        Format: {"instruction": ..., "input": "", "output": ...}
        """
        alpaca = []
        for item in dataset:
            alpaca.append({
                "instruction": f"[SYSTEM] {item['system']}\n\n[USER] {item['instruction']}",
                "input": "",
                "output": item["output"],
            })
        return alpaca

    def export_for_chat(self, dataset: list[dict]) -> list[dict]:
        """
        Convert to ChatML format for chat-tuned models.
        """
        chatml = []
        for item in dataset:
            chatml.append({
                "messages": [
                    {"role": "system", "content": item["system"]},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["output"]},
                ]
            })
        return chatml


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from mep_generator import MEPFamilyGenerator

    gen = MEPFamilyGenerator(seed=42)
    families = gen.generate_dataset(n_per_template=2)

    builder = MEPPromptBuilder(lang="ru", seed=42)
    dataset = builder.build_dataset(families, prompts_per_family=2)

    print(f"✅ Generated {len(dataset)} training pairs")
    print(f"   Domains: {sorted(set(d['metadata']['domain'] for d in dataset))}")
    print(f"   Levels: {sorted(set(d['metadata']['level'] for d in dataset))}")

    # Show one example
    example = dataset[0]
    print(f"\n{'='*60}")
    print(f"INSTRUCTION:\n{example['instruction']}")
    print(f"\nOUTPUT (truncated):\n{example['output'][:500]}...")

    # Export formats
    alpaca = builder.export_for_unsloth(dataset)
    chatml = builder.export_for_chat(dataset)
    print(f"\n✅ Alpaca format: {len(alpaca)} samples")
    print(f"✅ ChatML format: {len(chatml)} samples")
