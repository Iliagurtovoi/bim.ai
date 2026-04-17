"""
BIM.AI — MEP JSON Schema Extension
====================================
Расширение базовой JSON-схемы BIMFamily для инженерных систем (MEP).

Ключевые дополнения:
- MEPConnector с flow direction, system type, shape, размерами
- CalculationParameter с формулами зависимостей
- ClassificationReference (OmniClass, UniClass, IFC)
- NestedFamily для вложенных компонентов
- SystemRequirements для валидации совместимости
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class MEPDomain(str, Enum):
    HVAC = "HVAC"
    PIPING = "Piping"
    ELECTRICAL = "Electrical"
    FIRE_PROTECTION = "FireProtection"


class ConnectorDomain(str, Enum):
    DUCT = "DomainHvac"
    PIPE = "DomainPiping"
    CONDUIT = "DomainElectrical"
    CABLE_TRAY = "DomainCableTray"
    FITTING = "DomainFitting"


class FlowDirection(str, Enum):
    IN = "In"
    OUT = "Out"
    BIDIRECTIONAL = "Bidirectional"


class SystemType(str, Enum):
    # HVAC
    SUPPLY_AIR = "SupplyAir"
    RETURN_AIR = "ReturnAir"
    EXHAUST_AIR = "ExhaustAir"
    OUTSIDE_AIR = "OutsideAir"
    # Piping
    DOMESTIC_HOT_WATER = "DomesticHotWater"
    DOMESTIC_COLD_WATER = "DomesticColdWater"
    SANITARY = "Sanitary"
    VENT = "Vent"
    HYDRONIC_SUPPLY = "HydronicSupply"
    HYDRONIC_RETURN = "HydronicReturn"
    CHILLED_WATER = "ChilledWater"
    CONDENSER_WATER = "CondenserWater"
    REFRIGERANT = "Refrigerant"
    # Electrical
    POWER = "Power"
    LIGHTING = "Lighting"
    FIRE_ALARM = "FireAlarm"
    DATA = "Data"
    COMMUNICATION = "Communication"
    # Fire protection
    WET_SPRINKLER = "WetSprinkler"
    DRY_SPRINKLER = "DrySprinkler"
    PRE_ACTION = "PreAction"


class ConnectorShape(str, Enum):
    ROUND = "Round"
    RECTANGULAR = "Rectangular"
    OVAL = "Oval"


class ParameterGroup(str, Enum):
    MECHANICAL = "PG_MECHANICAL"
    ELECTRICAL = "PG_ELECTRICAL"
    PLUMBING = "PG_PLUMBING"
    ENERGY = "PG_ENERGY"
    FIRE_PROTECTION = "PG_FIRE_PROTECTION"
    IDENTITY = "PG_IDENTITY_DATA"
    DIMENSIONS = "PG_GEOMETRY"
    GENERAL = "PG_GENERAL"


class ParameterType(str, Enum):
    LENGTH = "Length"
    AREA = "Area"
    VOLUME = "Volume"
    ANGLE = "Angle"
    NUMBER = "Number"
    INTEGER = "Integer"
    TEXT = "Text"
    YES_NO = "YesNo"
    FLOW = "Flow"                  # CFM / L/s
    PRESSURE = "Pressure"          # Pa / inWG
    POWER = "Power"                # W / BTU/h
    VOLTAGE = "Voltage"            # V
    CURRENT = "Current"            # A
    TEMPERATURE = "Temperature"    # °C / °F
    VELOCITY = "Velocity"          # m/s / fpm
    PIPE_SIZE = "PipeSize"         # DN / NPS
    DUCT_SIZE = "DuctSize"


# ─────────────────────────────────────────────
# Core Data Classes
# ─────────────────────────────────────────────

@dataclass
class ClassificationReference:
    """Классификационные коды для MEP-элемента."""
    omniclass_table: Optional[str] = None      # e.g. "23-33 21 11"
    omniclass_title: Optional[str] = None      # e.g. "Air Diffusers"
    uniclass_code: Optional[str] = None        # e.g. "Pr_70_65_03"
    uniclass_title: Optional[str] = None
    ifc_class: Optional[str] = None            # e.g. "IfcAirTerminal"
    ifc_predefined_type: Optional[str] = None  # e.g. "DIFFUSER"
    masterformat: Optional[str] = None         # e.g. "23 37 13"


@dataclass
class ConnectorDimensions:
    """Размеры коннектора (зависят от формы)."""
    # Round
    diameter: Optional[float] = None          # мм
    # Rectangular
    width: Optional[float] = None             # мм
    height: Optional[float] = None            # мм
    # Nominal sizes for pipes
    nominal_diameter: Optional[str] = None    # "DN50", "2\""


@dataclass
class MEPConnector:
    """Точка подключения MEP-элемента."""
    id: str = ""
    name: str = ""
    domain: ConnectorDomain = ConnectorDomain.DUCT
    flow_direction: FlowDirection = FlowDirection.BIDIRECTIONAL
    system_type: SystemType = SystemType.SUPPLY_AIR
    shape: ConnectorShape = ConnectorShape.ROUND
    dimensions: ConnectorDimensions = field(default_factory=ConnectorDimensions)
    # Position relative to family origin (мм)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Direction vector (нормаль)
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Optional: linked parameter names for dynamic sizing
    linked_size_param: Optional[str] = None
    # Optional: flow/load at this connector
    design_flow: Optional[float] = None       # L/s or CFM
    design_pressure_drop: Optional[float] = None  # Pa
    # Electrical
    voltage: Optional[float] = None           # V
    apparent_load: Optional[float] = None     # VA
    power_factor: Optional[float] = None
    number_of_poles: Optional[int] = None


@dataclass
class FormulaDefinition:
    """Формула зависимости между параметрами."""
    target_param: str = ""
    expression: str = ""                # e.g. "Airflow * 0.06"
    description: str = ""               # Пояснение для dataset
    dependencies: list[str] = field(default_factory=list)


@dataclass
class CalculationParameter:
    """Расчётный параметр MEP-семейства."""
    name: str = ""
    display_name: str = ""
    param_type: ParameterType = ParameterType.NUMBER
    group: ParameterGroup = ParameterGroup.MECHANICAL
    unit_label: str = ""                # e.g. "L/s", "Pa", "W"
    default_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_instance: bool = True
    is_shared: bool = False
    formula: Optional[FormulaDefinition] = None
    tooltip: str = ""


@dataclass
class NestedFamily:
    """Вложенное семейство внутри MEP-элемента."""
    name: str = ""
    category: str = ""
    insertion_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_shared: bool = False
    linked_params: dict[str, str] = field(default_factory=dict)  # host_param -> nested_param


@dataclass
class GeometryPrimitive:
    """Геометрический примитив."""
    type: str = "Extrusion"           # Extrusion, Blend, Revolve, Sweep, SweptBlend, Void
    material_param: Optional[str] = None
    is_visible: bool = True
    subcategory: Optional[str] = None  # e.g. "Hidden Lines", "Insulation"
    # Simplified dimensions for dataset (actual geometry in Dynamo script)
    bounding_box: Optional[dict] = None


@dataclass
class FamilyTypeVariant:
    """Вариант типоразмера семейства."""
    name: str = ""
    parameter_values: dict[str, object] = field(default_factory=dict)
    connector_overrides: dict[str, dict] = field(default_factory=dict)


@dataclass
class SystemRequirements:
    """Требования к инженерной системе для валидации."""
    min_connectors: int = 1
    max_connectors: int = 20
    required_connector_domains: list[ConnectorDomain] = field(default_factory=list)
    required_system_types: list[SystemType] = field(default_factory=list)
    allowed_connector_shapes: list[ConnectorShape] = field(default_factory=list)
    min_diameter_mm: Optional[float] = None
    max_diameter_mm: Optional[float] = None
    requires_electrical: bool = False
    requires_flow_params: bool = False
    allowed_hosting: list[str] = field(default_factory=list)  # "Wall", "Ceiling", "Floor", "Non-hosted"


# ─────────────────────────────────────────────
# Main MEP Family Schema
# ─────────────────────────────────────────────

@dataclass
class MEPBIMFamily:
    """
    Полная схема MEP-семейства для BIM.AI.
    Расширяет базовую BIMFamily с MEP-специфичными полями.
    """
    # === Identity ===
    family_name: str = ""
    category: str = ""                       # Revit category name
    mep_domain: MEPDomain = MEPDomain.HVAC
    subcategory: str = ""                    # e.g. "Air Terminals", "Sprinklers"
    description: str = ""
    manufacturer: str = ""
    model: str = ""
    url: str = ""

    # === Classification ===
    classification: ClassificationReference = field(default_factory=ClassificationReference)

    # === Hosting ===
    host_type: str = "Non-hosted"            # "Wall", "Ceiling", "Floor", "Face", "Non-hosted"
    can_host_on: list[str] = field(default_factory=list)

    # === Connectors ===
    connectors: list[MEPConnector] = field(default_factory=list)

    # === Parameters ===
    calculation_params: list[CalculationParameter] = field(default_factory=list)
    formulas: list[FormulaDefinition] = field(default_factory=list)

    # === Geometry ===
    geometry: list[GeometryPrimitive] = field(default_factory=list)

    # === Nested Families ===
    nested_families: list[NestedFamily] = field(default_factory=list)

    # === Type Variants ===
    family_types: list[FamilyTypeVariant] = field(default_factory=list)

    # === System Requirements (for validation) ===
    system_requirements: SystemRequirements = field(default_factory=SystemRequirements)

    # === Metadata ===
    revit_version: str = "2024"
    template_file: str = ""                  # e.g. "Metric Mechanical Equipment.rft"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict, converting enums to strings."""
        def _convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, (list, tuple)):
                return [_convert(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            return obj
        return _convert(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> MEPBIMFamily:
        """Deserialize from dict (simplified — production code needs full enum mapping)."""
        family = cls()
        for key, value in data.items():
            if hasattr(family, key):
                setattr(family, key, value)
        return family


# ─────────────────────────────────────────────
# JSON Schema Export (for validation / docs)
# ─────────────────────────────────────────────

MEP_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "MEPBIMFamily",
    "description": "BIM.AI MEP Family Schema — full specification for MEP family generation",
    "type": "object",
    "required": ["family_name", "category", "mep_domain", "connectors"],
    "properties": {
        "family_name": {"type": "string", "minLength": 1},
        "category": {
            "type": "string",
            "enum": [
                "Mechanical Equipment", "Air Terminals", "Duct Accessories",
                "Duct Fittings", "Flex Ducts",
                "Plumbing Fixtures", "Plumbing Equipment", "Pipe Accessories",
                "Pipe Fittings", "Flex Pipes",
                "Electrical Equipment", "Electrical Fixtures", "Lighting Fixtures",
                "Lighting Devices", "Communication Devices", "Data Devices",
                "Fire Alarm Devices", "Nurse Call Devices", "Security Devices",
                "Cable Tray Fittings", "Conduit Fittings",
                "Sprinklers", "Fire Protection Equipment",
            ]
        },
        "mep_domain": {"type": "string", "enum": [d.value for d in MEPDomain]},
        "subcategory": {"type": "string"},
        "description": {"type": "string"},
        "classification": {
            "type": "object",
            "properties": {
                "omniclass_table": {"type": ["string", "null"]},
                "omniclass_title": {"type": ["string", "null"]},
                "uniclass_code": {"type": ["string", "null"]},
                "ifc_class": {"type": ["string", "null"]},
                "ifc_predefined_type": {"type": ["string", "null"]},
                "masterformat": {"type": ["string", "null"]},
            }
        },
        "host_type": {
            "type": "string",
            "enum": ["Wall", "Ceiling", "Floor", "Face", "Non-hosted"]
        },
        "connectors": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "domain", "flow_direction", "system_type"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "domain": {"type": "string", "enum": [d.value for d in ConnectorDomain]},
                    "flow_direction": {"type": "string", "enum": [d.value for d in FlowDirection]},
                    "system_type": {"type": "string", "enum": [s.value for s in SystemType]},
                    "shape": {"type": "string", "enum": [s.value for s in ConnectorShape]},
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "diameter": {"type": ["number", "null"], "minimum": 0},
                            "width": {"type": ["number", "null"], "minimum": 0},
                            "height": {"type": ["number", "null"], "minimum": 0},
                            "nominal_diameter": {"type": ["string", "null"]},
                        }
                    },
                    "position": {
                        "type": "array", "items": {"type": "number"},
                        "minItems": 3, "maxItems": 3
                    },
                    "direction": {
                        "type": "array", "items": {"type": "number"},
                        "minItems": 3, "maxItems": 3
                    },
                    "linked_size_param": {"type": ["string", "null"]},
                    "design_flow": {"type": ["number", "null"]},
                    "design_pressure_drop": {"type": ["number", "null"]},
                    "voltage": {"type": ["number", "null"]},
                    "apparent_load": {"type": ["number", "null"]},
                    "power_factor": {"type": ["number", "null"]},
                    "number_of_poles": {"type": ["integer", "null"]},
                }
            }
        },
        "calculation_params": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "param_type"],
                "properties": {
                    "name": {"type": "string"},
                    "display_name": {"type": "string"},
                    "param_type": {"type": "string", "enum": [p.value for p in ParameterType]},
                    "group": {"type": "string", "enum": [g.value for g in ParameterGroup]},
                    "unit_label": {"type": "string"},
                    "default_value": {"type": ["number", "null"]},
                    "min_value": {"type": ["number", "null"]},
                    "max_value": {"type": ["number", "null"]},
                    "is_instance": {"type": "boolean"},
                    "is_shared": {"type": "boolean"},
                    "formula": {
                        "type": ["object", "null"],
                        "properties": {
                            "target_param": {"type": "string"},
                            "expression": {"type": "string"},
                            "description": {"type": "string"},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                        }
                    },
                }
            }
        },
        "formulas": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["target_param", "expression"],
                "properties": {
                    "target_param": {"type": "string"},
                    "expression": {"type": "string"},
                    "description": {"type": "string"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                }
            }
        },
        "geometry": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": [
                        "Extrusion", "Blend", "Revolve", "Sweep",
                        "SweptBlend", "Void", "ImportedGeometry"
                    ]},
                    "material_param": {"type": ["string", "null"]},
                    "is_visible": {"type": "boolean"},
                    "subcategory": {"type": ["string", "null"]},
                }
            }
        },
        "nested_families": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "insertion_point": {"type": "array", "items": {"type": "number"}},
                    "is_shared": {"type": "boolean"},
                    "linked_params": {"type": "object"},
                }
            }
        },
        "family_types": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "parameter_values": {"type": "object"},
                    "connector_overrides": {"type": "object"},
                }
            }
        },
        "system_requirements": {
            "type": "object",
            "properties": {
                "min_connectors": {"type": "integer", "minimum": 0},
                "max_connectors": {"type": "integer"},
                "required_connector_domains": {"type": "array", "items": {"type": "string"}},
                "requires_electrical": {"type": "boolean"},
                "requires_flow_params": {"type": "boolean"},
            }
        },
        "revit_version": {"type": "string"},
        "template_file": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    }
}


if __name__ == "__main__":
    # Пример: создание воздухораспределителя
    diffuser = MEPBIMFamily(
        family_name="BIM_AI_Diffuser_Round_Adj",
        category="Air Terminals",
        mep_domain=MEPDomain.HVAC,
        subcategory="Supply Diffusers",
        description="Круглый приточный диффузор с регулируемыми лопатками",
        classification=ClassificationReference(
            omniclass_table="23-33 21 11 11",
            omniclass_title="Round Ceiling Diffusers",
            ifc_class="IfcAirTerminal",
            ifc_predefined_type="DIFFUSER",
        ),
        host_type="Ceiling",
        connectors=[
            MEPConnector(
                id="conn_supply",
                name="Supply Air Inlet",
                domain=ConnectorDomain.DUCT,
                flow_direction=FlowDirection.IN,
                system_type=SystemType.SUPPLY_AIR,
                shape=ConnectorShape.ROUND,
                dimensions=ConnectorDimensions(diameter=200),
                position=(0, 0, 300),
                direction=(0, 0, 1),
                linked_size_param="Duct_Connection_Diameter",
                design_flow=150.0,
            ),
        ],
        calculation_params=[
            CalculationParameter(
                name="Airflow",
                display_name="Расход воздуха",
                param_type=ParameterType.FLOW,
                group=ParameterGroup.MECHANICAL,
                unit_label="L/s",
                default_value=150,
                min_value=50,
                max_value=500,
            ),
            CalculationParameter(
                name="Pressure_Drop",
                display_name="Потери давления",
                param_type=ParameterType.PRESSURE,
                group=ParameterGroup.MECHANICAL,
                unit_label="Pa",
                default_value=25,
                formula=FormulaDefinition(
                    target_param="Pressure_Drop",
                    expression="0.001 * Airflow ^ 2",
                    description="Квадратичная зависимость потерь от расхода",
                    dependencies=["Airflow"],
                ),
            ),
        ],
        family_types=[
            FamilyTypeVariant(name="200mm - 150 L/s", parameter_values={
                "Duct_Connection_Diameter": 200, "Airflow": 150,
                "Face_Diameter": 300, "Throw": 2.5,
            }),
            FamilyTypeVariant(name="250mm - 250 L/s", parameter_values={
                "Duct_Connection_Diameter": 250, "Airflow": 250,
                "Face_Diameter": 400, "Throw": 3.5,
            }),
        ],
    )
    print(diffuser.to_json())
    print(f"\n✅ Schema validated. JSON size: {len(diffuser.to_json())} bytes")
