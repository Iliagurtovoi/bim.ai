"""
BIM.AI — Synthetic MEP Family Generator
=========================================
Генерирует реалистичные MEP-семейства для обучающего датасета.
Покрывает 4 домена: HVAC, Piping, Electrical, Fire Protection.

Каждый генератор создаёт семейства с:
- Реалистичными типоразмерами (стандартные ряды DN, мм, дюймы)
- Правильными коннекторами (позиции, направления, системы)
- Расчётными параметрами с формулами
- Классификацией OmniClass / IFC
- Вложенными семействами где уместно
"""

from __future__ import annotations
import random
import math
import uuid
from dataclasses import dataclass, field
from typing import Callable

from mep_schema import (
    MEPBIMFamily, MEPDomain, MEPConnector, ConnectorDomain,
    FlowDirection, SystemType, ConnectorShape, ConnectorDimensions,
    CalculationParameter, ParameterType, ParameterGroup,
    FormulaDefinition, ClassificationReference, NestedFamily,
    FamilyTypeVariant, GeometryPrimitive, SystemRequirements,
)


# ─────────────────────────────────────────────
# Standard Engineering Sizes
# ─────────────────────────────────────────────

DUCT_ROUND_DIAMETERS = [100, 125, 150, 160, 200, 250, 300, 315, 355, 400, 450, 500, 560, 630, 710, 800, 900, 1000, 1120, 1250]
DUCT_RECT_WIDTHS = [200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500]
DUCT_RECT_HEIGHTS = [150, 200, 250, 300, 400, 500, 600, 800]

PIPE_DN = [15, 20, 25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300]
PIPE_DN_LABELS = {d: f"DN{d}" for d in PIPE_DN}

CONDUIT_DIAMETERS = [16, 20, 25, 32, 40, 50, 63]
CABLE_TRAY_WIDTHS = [100, 150, 200, 300, 400, 500, 600]

VOLTAGES = [220, 230, 380, 400, 690]
AMPERE_RATINGS = [6, 10, 16, 20, 25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630]

SPRINKLER_K_FACTORS = [57, 80, 115, 161, 202, 242, 363]  # metric K-factor

MANUFACTURERS = {
    "HVAC": ["Systemair", "Trox", "Swegon", "Lindab", "Fläkt Group", "Daikin", "Carrier", "Trane"],
    "Piping": ["Grundfos", "Wilo", "Giacomini", "Danfoss", "Flamco", "Valtec", "Rehau"],
    "Electrical": ["ABB", "Schneider Electric", "Legrand", "Siemens", "Eaton", "Hager", "IEK"],
    "FireProtection": ["Viking", "Tyco", "Victaulic", "Minimax", "Hochiki", "Bosch Security"],
}


# ─────────────────────────────────────────────
# Equipment Catalog — defines what can be generated
# ─────────────────────────────────────────────

@dataclass
class EquipmentTemplate:
    """Template for generating a specific equipment type."""
    name_pattern: str
    category: str
    mep_domain: MEPDomain
    subcategory: str
    host_type: str
    ifc_class: str
    ifc_predefined_type: str
    omniclass: str
    template_file: str
    tags: list[str]
    connector_specs: list[dict]       # Specs for each connector
    param_specs: list[dict]           # Specs for calculation params
    formula_specs: list[dict] = field(default_factory=list)
    nested_specs: list[dict] = field(default_factory=list)
    type_generator: str = ""          # Name of type variant generator
    geometry_types: list[str] = field(default_factory=lambda: ["Extrusion"])


# ─────────────────────────────────────────────
# HVAC Equipment Templates
# ─────────────────────────────────────────────

HVAC_TEMPLATES = [
    # Round Supply Diffuser
    EquipmentTemplate(
        name_pattern="BIM_AI_RoundDiffuser_{size}",
        category="Air Terminals",
        mep_domain=MEPDomain.HVAC,
        subcategory="Supply Diffusers",
        host_type="Ceiling",
        ifc_class="IfcAirTerminal",
        ifc_predefined_type="DIFFUSER",
        omniclass="23-33 21 11 11",
        template_file="Metric Air Terminal.rft",
        tags=["HVAC", "air_terminal", "diffuser", "round", "ceiling"],
        connector_specs=[{
            "domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.IN,
            "system": SystemType.SUPPLY_AIR, "shape": ConnectorShape.ROUND,
            "size_param": "Duct_Diameter", "pos_offset": (0, 0, 1),
            "dir": (0, 0, 1),
        }],
        param_specs=[
            {"name": "Airflow", "type": ParameterType.FLOW, "group": ParameterGroup.MECHANICAL,
             "unit": "L/s", "range": (50, 600)},
            {"name": "Pressure_Drop", "type": ParameterType.PRESSURE, "group": ParameterGroup.MECHANICAL,
             "unit": "Pa", "range": (10, 80)},
            {"name": "Throw", "type": ParameterType.LENGTH, "group": ParameterGroup.MECHANICAL,
             "unit": "m", "range": (1.0, 8.0)},
            {"name": "Noise_Level", "type": ParameterType.NUMBER, "group": ParameterGroup.MECHANICAL,
             "unit": "NC", "range": (15, 40)},
            {"name": "Face_Diameter", "type": ParameterType.LENGTH, "group": ParameterGroup.DIMENSIONS,
             "unit": "mm", "range": (200, 800)},
        ],
        formula_specs=[
            {"target": "Pressure_Drop", "expr": "0.0008 * Airflow ^ 2 + 5",
             "deps": ["Airflow"], "desc": "Квадратичная зависимость от расхода"},
            {"target": "Face_Diameter", "expr": "Duct_Diameter + 100",
             "deps": ["Duct_Diameter"], "desc": "Лицевая панель = duct + 100мм"},
        ],
        type_generator="diffuser_round",
    ),
    # Rectangular Grille
    EquipmentTemplate(
        name_pattern="BIM_AI_RectGrille_{type}_{size}",
        category="Air Terminals",
        mep_domain=MEPDomain.HVAC,
        subcategory="Supply Grilles",
        host_type="Wall",
        ifc_class="IfcAirTerminal",
        ifc_predefined_type="GRILLE",
        omniclass="23-33 21 11 15",
        template_file="Metric Air Terminal.rft",
        tags=["HVAC", "air_terminal", "grille", "rectangular", "wall"],
        connector_specs=[{
            "domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.IN,
            "system": SystemType.SUPPLY_AIR, "shape": ConnectorShape.RECTANGULAR,
            "size_param": "Duct_Width", "pos_offset": (0, 0, -1),
            "dir": (0, 0, -1),
        }],
        param_specs=[
            {"name": "Airflow", "type": ParameterType.FLOW, "group": ParameterGroup.MECHANICAL,
             "unit": "L/s", "range": (30, 400)},
            {"name": "Pressure_Drop", "type": ParameterType.PRESSURE, "group": ParameterGroup.MECHANICAL,
             "unit": "Pa", "range": (8, 60)},
            {"name": "Effective_Area", "type": ParameterType.AREA, "group": ParameterGroup.MECHANICAL,
             "unit": "m²", "range": (0.01, 0.5)},
        ],
        type_generator="grille_rect",
    ),
    # Fan Coil Unit (4-pipe)
    EquipmentTemplate(
        name_pattern="BIM_AI_FCU_{config}_{capacity}kW",
        category="Mechanical Equipment",
        mep_domain=MEPDomain.HVAC,
        subcategory="Fan Coil Units",
        host_type="Ceiling",
        ifc_class="IfcUnitaryEquipment",
        ifc_predefined_type="AIRHANDLER",
        omniclass="23-33 19 11",
        template_file="Metric Mechanical Equipment.rft",
        tags=["HVAC", "FCU", "fan_coil", "4pipe", "ceiling"],
        connector_specs=[
            {"domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.OUT,
             "system": SystemType.SUPPLY_AIR, "shape": ConnectorShape.ROUND,
             "size_param": "Supply_Duct_Diameter", "pos_offset": (1, 0, 0), "dir": (1, 0, 0)},
            {"domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.IN,
             "system": SystemType.RETURN_AIR, "shape": ConnectorShape.ROUND,
             "size_param": "Return_Duct_Diameter", "pos_offset": (-1, 0, 0), "dir": (-1, 0, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.IN,
             "system": SystemType.CHILLED_WATER, "shape": ConnectorShape.ROUND,
             "size_param": "CHW_Pipe_Size", "pos_offset": (0, -1, 0), "dir": (0, -1, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.OUT,
             "system": SystemType.CHILLED_WATER, "shape": ConnectorShape.ROUND,
             "size_param": "CHW_Return_Pipe_Size", "pos_offset": (0, -1, 0.1), "dir": (0, -1, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.IN,
             "system": SystemType.HYDRONIC_SUPPLY, "shape": ConnectorShape.ROUND,
             "size_param": "HW_Pipe_Size", "pos_offset": (0, 1, 0), "dir": (0, 1, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.OUT,
             "system": SystemType.HYDRONIC_RETURN, "shape": ConnectorShape.ROUND,
             "size_param": "HW_Return_Pipe_Size", "pos_offset": (0, 1, 0.1), "dir": (0, 1, 0)},
        ],
        param_specs=[
            {"name": "Cooling_Capacity", "type": ParameterType.POWER, "group": ParameterGroup.MECHANICAL,
             "unit": "kW", "range": (1.5, 15)},
            {"name": "Heating_Capacity", "type": ParameterType.POWER, "group": ParameterGroup.MECHANICAL,
             "unit": "kW", "range": (2, 20)},
            {"name": "Airflow", "type": ParameterType.FLOW, "group": ParameterGroup.MECHANICAL,
             "unit": "L/s", "range": (100, 1200)},
            {"name": "Fan_Power", "type": ParameterType.POWER, "group": ParameterGroup.ELECTRICAL,
             "unit": "W", "range": (30, 300)},
            {"name": "Noise_Level", "type": ParameterType.NUMBER, "group": ParameterGroup.MECHANICAL,
             "unit": "dB(A)", "range": (25, 50)},
            {"name": "ESP", "type": ParameterType.PRESSURE, "group": ParameterGroup.MECHANICAL,
             "unit": "Pa", "range": (30, 150)},
        ],
        formula_specs=[
            {"target": "Airflow", "expr": "Cooling_Capacity * 80",
             "deps": ["Cooling_Capacity"], "desc": "~80 L/s на kW холода"},
            {"target": "Fan_Power", "expr": "Airflow * ESP / 500",
             "deps": ["Airflow", "ESP"], "desc": "Мощность вентилятора = Q*P/eta"},
        ],
        type_generator="fcu_4pipe",
    ),
    # VAV Box
    EquipmentTemplate(
        name_pattern="BIM_AI_VAV_{type}_{size}",
        category="Duct Accessories",
        mep_domain=MEPDomain.HVAC,
        subcategory="VAV Boxes",
        host_type="Non-hosted",
        ifc_class="IfcDamper",
        ifc_predefined_type="CONTROLDAMPER",
        omniclass="23-33 23 11",
        template_file="Metric Duct Accessory.rft",
        tags=["HVAC", "VAV", "variable_air_volume", "terminal_unit"],
        connector_specs=[
            {"domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.IN,
             "system": SystemType.SUPPLY_AIR, "shape": ConnectorShape.ROUND,
             "size_param": "Inlet_Diameter", "pos_offset": (-1, 0, 0), "dir": (-1, 0, 0)},
            {"domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.OUT,
             "system": SystemType.SUPPLY_AIR, "shape": ConnectorShape.ROUND,
             "size_param": "Outlet_Diameter", "pos_offset": (1, 0, 0), "dir": (1, 0, 0)},
        ],
        param_specs=[
            {"name": "Min_Airflow", "type": ParameterType.FLOW, "group": ParameterGroup.MECHANICAL,
             "unit": "L/s", "range": (20, 300)},
            {"name": "Max_Airflow", "type": ParameterType.FLOW, "group": ParameterGroup.MECHANICAL,
             "unit": "L/s", "range": (50, 1500)},
            {"name": "Pressure_Drop", "type": ParameterType.PRESSURE, "group": ParameterGroup.MECHANICAL,
             "unit": "Pa", "range": (20, 120)},
            {"name": "Has_Reheat", "type": ParameterType.YES_NO, "group": ParameterGroup.MECHANICAL,
             "unit": "", "range": (0, 1)},
            {"name": "Reheat_Capacity", "type": ParameterType.POWER, "group": ParameterGroup.MECHANICAL,
             "unit": "kW", "range": (0, 10)},
        ],
        formula_specs=[
            {"target": "Min_Airflow", "expr": "Max_Airflow * 0.3",
             "deps": ["Max_Airflow"], "desc": "Мин расход = 30% макс"},
        ],
        type_generator="vav_box",
    ),
    # AHU (Air Handling Unit)
    EquipmentTemplate(
        name_pattern="BIM_AI_AHU_{config}_{airflow}",
        category="Mechanical Equipment",
        mep_domain=MEPDomain.HVAC,
        subcategory="Air Handling Units",
        host_type="Non-hosted",
        ifc_class="IfcAirToAirHeatRecovery",
        ifc_predefined_type="NOTDEFINED",
        omniclass="23-33 11 11",
        template_file="Metric Mechanical Equipment.rft",
        tags=["HVAC", "AHU", "air_handling_unit", "central"],
        connector_specs=[
            {"domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.OUT,
             "system": SystemType.SUPPLY_AIR, "shape": ConnectorShape.RECTANGULAR,
             "size_param": "SA_Duct_Width", "pos_offset": (1, 0, 0.5), "dir": (1, 0, 0)},
            {"domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.IN,
             "system": SystemType.RETURN_AIR, "shape": ConnectorShape.RECTANGULAR,
             "size_param": "RA_Duct_Width", "pos_offset": (1, 0, -0.5), "dir": (1, 0, 0)},
            {"domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.IN,
             "system": SystemType.OUTSIDE_AIR, "shape": ConnectorShape.RECTANGULAR,
             "size_param": "OA_Duct_Width", "pos_offset": (-1, 0, 0.5), "dir": (-1, 0, 0)},
            {"domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.OUT,
             "system": SystemType.EXHAUST_AIR, "shape": ConnectorShape.RECTANGULAR,
             "size_param": "EA_Duct_Width", "pos_offset": (-1, 0, -0.5), "dir": (-1, 0, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.BIDIRECTIONAL,
             "system": SystemType.CHILLED_WATER, "shape": ConnectorShape.ROUND,
             "size_param": "CHW_Pipe", "pos_offset": (0, -1, 0), "dir": (0, -1, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.BIDIRECTIONAL,
             "system": SystemType.HYDRONIC_SUPPLY, "shape": ConnectorShape.ROUND,
             "size_param": "HW_Pipe", "pos_offset": (0, -1, 0.2), "dir": (0, -1, 0)},
        ],
        param_specs=[
            {"name": "Supply_Airflow", "type": ParameterType.FLOW, "group": ParameterGroup.MECHANICAL,
             "unit": "m³/h", "range": (1000, 80000)},
            {"name": "ESP", "type": ParameterType.PRESSURE, "group": ParameterGroup.MECHANICAL,
             "unit": "Pa", "range": (200, 1500)},
            {"name": "Cooling_Capacity", "type": ParameterType.POWER, "group": ParameterGroup.MECHANICAL,
             "unit": "kW", "range": (10, 500)},
            {"name": "Heating_Capacity", "type": ParameterType.POWER, "group": ParameterGroup.MECHANICAL,
             "unit": "kW", "range": (10, 400)},
            {"name": "Fan_Power", "type": ParameterType.POWER, "group": ParameterGroup.ELECTRICAL,
             "unit": "kW", "range": (1, 75)},
            {"name": "Heat_Recovery_Efficiency", "type": ParameterType.NUMBER, "group": ParameterGroup.MECHANICAL,
             "unit": "%", "range": (50, 90)},
            {"name": "Filter_Class", "type": ParameterType.TEXT, "group": ParameterGroup.MECHANICAL,
             "unit": "", "range": None},
        ],
        type_generator="ahu",
    ),
]


# ─────────────────────────────────────────────
# Piping Equipment Templates
# ─────────────────────────────────────────────

PIPING_TEMPLATES = [
    # Plumbing Fixture — Washbasin
    EquipmentTemplate(
        name_pattern="BIM_AI_Washbasin_{type}_{size}",
        category="Plumbing Fixtures",
        mep_domain=MEPDomain.PIPING,
        subcategory="Washbasins",
        host_type="Wall",
        ifc_class="IfcSanitaryTerminal",
        ifc_predefined_type="WASHHANDBASIN",
        omniclass="23-31 21 11 11",
        template_file="Metric Plumbing Fixture.rft",
        tags=["plumbing", "fixture", "washbasin", "sanitary"],
        connector_specs=[
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.IN,
             "system": SystemType.DOMESTIC_COLD_WATER, "shape": ConnectorShape.ROUND,
             "size_param": "CW_Connection", "pos_offset": (0, -0.5, -0.3), "dir": (0, -1, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.IN,
             "system": SystemType.DOMESTIC_HOT_WATER, "shape": ConnectorShape.ROUND,
             "size_param": "HW_Connection", "pos_offset": (0, -0.5, -0.3), "dir": (0, -1, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.OUT,
             "system": SystemType.SANITARY, "shape": ConnectorShape.ROUND,
             "size_param": "Drain_Connection", "pos_offset": (0, -0.5, -0.5), "dir": (0, 0, -1)},
        ],
        param_specs=[
            {"name": "Flow_Rate", "type": ParameterType.FLOW, "group": ParameterGroup.PLUMBING,
             "unit": "L/s", "range": (0.1, 0.3)},
            {"name": "Fixture_Units", "type": ParameterType.NUMBER, "group": ParameterGroup.PLUMBING,
             "unit": "FU", "range": (0.5, 2)},
            {"name": "Basin_Width", "type": ParameterType.LENGTH, "group": ParameterGroup.DIMENSIONS,
             "unit": "mm", "range": (400, 700)},
            {"name": "Basin_Depth", "type": ParameterType.LENGTH, "group": ParameterGroup.DIMENSIONS,
             "unit": "mm", "range": (300, 550)},
            {"name": "Mounting_Height", "type": ParameterType.LENGTH, "group": ParameterGroup.DIMENSIONS,
             "unit": "mm", "range": (750, 900)},
        ],
        type_generator="washbasin",
    ),
    # Pump
    EquipmentTemplate(
        name_pattern="BIM_AI_Pump_{type}_{flow}",
        category="Plumbing Equipment",
        mep_domain=MEPDomain.PIPING,
        subcategory="Circulation Pumps",
        host_type="Non-hosted",
        ifc_class="IfcPump",
        ifc_predefined_type="CIRCULATOR",
        omniclass="23-29 11 11",
        template_file="Metric Plumbing Equipment.rft",
        tags=["piping", "pump", "circulation", "mechanical"],
        connector_specs=[
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.IN,
             "system": SystemType.HYDRONIC_SUPPLY, "shape": ConnectorShape.ROUND,
             "size_param": "Inlet_DN", "pos_offset": (-1, 0, 0), "dir": (-1, 0, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.OUT,
             "system": SystemType.HYDRONIC_SUPPLY, "shape": ConnectorShape.ROUND,
             "size_param": "Outlet_DN", "pos_offset": (0, 0, 1), "dir": (0, 0, 1)},
        ],
        param_specs=[
            {"name": "Flow_Rate", "type": ParameterType.FLOW, "group": ParameterGroup.PLUMBING,
             "unit": "m³/h", "range": (0.5, 100)},
            {"name": "Head", "type": ParameterType.PRESSURE, "group": ParameterGroup.PLUMBING,
             "unit": "m", "range": (2, 30)},
            {"name": "Motor_Power", "type": ParameterType.POWER, "group": ParameterGroup.ELECTRICAL,
             "unit": "kW", "range": (0.05, 30)},
            {"name": "RPM", "type": ParameterType.NUMBER, "group": ParameterGroup.MECHANICAL,
             "unit": "rpm", "range": (1000, 3000)},
        ],
        formula_specs=[
            {"target": "Motor_Power", "expr": "(Flow_Rate * Head * 9.81) / (3600 * 0.7)",
             "deps": ["Flow_Rate", "Head"], "desc": "P = Q*H*ρg/η, η=0.7"},
        ],
        type_generator="pump",
    ),
    # Water Heater
    EquipmentTemplate(
        name_pattern="BIM_AI_WaterHeater_{type}_{volume}L",
        category="Plumbing Equipment",
        mep_domain=MEPDomain.PIPING,
        subcategory="Water Heaters",
        host_type="Non-hosted",
        ifc_class="IfcBoiler",
        ifc_predefined_type="WATER",
        omniclass="23-29 13 11",
        template_file="Metric Plumbing Equipment.rft",
        tags=["piping", "water_heater", "hot_water", "storage"],
        connector_specs=[
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.IN,
             "system": SystemType.DOMESTIC_COLD_WATER, "shape": ConnectorShape.ROUND,
             "size_param": "CW_Inlet_DN", "pos_offset": (0, -1, -0.3), "dir": (0, -1, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.OUT,
             "system": SystemType.DOMESTIC_HOT_WATER, "shape": ConnectorShape.ROUND,
             "size_param": "HW_Outlet_DN", "pos_offset": (0, -1, 0.3), "dir": (0, -1, 0)},
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.OUT,
             "system": SystemType.HYDRONIC_RETURN, "shape": ConnectorShape.ROUND,
             "size_param": "Recirc_DN", "pos_offset": (0, -1, 0), "dir": (0, -1, 0)},
        ],
        param_specs=[
            {"name": "Volume", "type": ParameterType.VOLUME, "group": ParameterGroup.PLUMBING,
             "unit": "L", "range": (30, 1000)},
            {"name": "Heating_Power", "type": ParameterType.POWER, "group": ParameterGroup.MECHANICAL,
             "unit": "kW", "range": (2, 50)},
            {"name": "Max_Temperature", "type": ParameterType.TEMPERATURE, "group": ParameterGroup.PLUMBING,
             "unit": "°C", "range": (55, 90)},
        ],
        type_generator="water_heater",
    ),
]


# ─────────────────────────────────────────────
# Electrical Equipment Templates
# ─────────────────────────────────────────────

ELECTRICAL_TEMPLATES = [
    # Distribution Panel
    EquipmentTemplate(
        name_pattern="BIM_AI_Panel_{type}_{amps}A",
        category="Electrical Equipment",
        mep_domain=MEPDomain.ELECTRICAL,
        subcategory="Distribution Panels",
        host_type="Wall",
        ifc_class="IfcElectricDistributionBoard",
        ifc_predefined_type="DISTRIBUTIONBOARD",
        omniclass="23-35 11 11",
        template_file="Metric Electrical Equipment.rft",
        tags=["electrical", "panel", "distribution", "switchboard"],
        connector_specs=[
            {"domain": ConnectorDomain.CONDUIT, "flow_dir": FlowDirection.BIDIRECTIONAL,
             "system": SystemType.POWER, "shape": ConnectorShape.ROUND,
             "size_param": "Main_Feed_Size", "pos_offset": (0, 0, -1), "dir": (0, 0, -1),
             "voltage": 400, "poles": 3},
        ],
        param_specs=[
            {"name": "Main_Bus_Rating", "type": ParameterType.CURRENT, "group": ParameterGroup.ELECTRICAL,
             "unit": "A", "range": (63, 630)},
            {"name": "Short_Circuit_Rating", "type": ParameterType.NUMBER, "group": ParameterGroup.ELECTRICAL,
             "unit": "kA", "range": (10, 65)},
            {"name": "Voltage", "type": ParameterType.VOLTAGE, "group": ParameterGroup.ELECTRICAL,
             "unit": "V", "range": (230, 400)},
            {"name": "Number_of_Ways", "type": ParameterType.INTEGER, "group": ParameterGroup.ELECTRICAL,
             "unit": "", "range": (6, 72)},
            {"name": "Total_Connected_Load", "type": ParameterType.POWER, "group": ParameterGroup.ELECTRICAL,
             "unit": "kW", "range": (5, 250)},
        ],
        type_generator="panel",
    ),
    # Power Outlet
    EquipmentTemplate(
        name_pattern="BIM_AI_Outlet_{type}_{config}",
        category="Electrical Fixtures",
        mep_domain=MEPDomain.ELECTRICAL,
        subcategory="Power Outlets",
        host_type="Wall",
        ifc_class="IfcOutlet",
        ifc_predefined_type="POWEROUTLET",
        omniclass="23-35 17 11",
        template_file="Metric Electrical Fixture - Wall Hosted.rft",
        tags=["electrical", "outlet", "socket", "power"],
        connector_specs=[
            {"domain": ConnectorDomain.CONDUIT, "flow_dir": FlowDirection.IN,
             "system": SystemType.POWER, "shape": ConnectorShape.ROUND,
             "size_param": "Conduit_Size", "pos_offset": (0, 0, -1), "dir": (0, 0, -1),
             "voltage": 230, "poles": 1},
        ],
        param_specs=[
            {"name": "Voltage", "type": ParameterType.VOLTAGE, "group": ParameterGroup.ELECTRICAL,
             "unit": "V", "range": (220, 230)},
            {"name": "Amperage", "type": ParameterType.CURRENT, "group": ParameterGroup.ELECTRICAL,
             "unit": "A", "range": (10, 32)},
            {"name": "Number_of_Gangs", "type": ParameterType.INTEGER, "group": ParameterGroup.ELECTRICAL,
             "unit": "", "range": (1, 4)},
            {"name": "Mounting_Height", "type": ParameterType.LENGTH, "group": ParameterGroup.DIMENSIONS,
             "unit": "mm", "range": (300, 1200)},
        ],
        type_generator="outlet",
    ),
    # Light Fixture
    EquipmentTemplate(
        name_pattern="BIM_AI_Light_{type}_{wattage}W",
        category="Lighting Fixtures",
        mep_domain=MEPDomain.ELECTRICAL,
        subcategory="LED Fixtures",
        host_type="Ceiling",
        ifc_class="IfcLightFixture",
        ifc_predefined_type="NOTDEFINED",
        omniclass="23-35 15 11",
        template_file="Metric Lighting Fixture (Ceiling Hosted).rft",
        tags=["electrical", "lighting", "LED", "ceiling"],
        connector_specs=[
            {"domain": ConnectorDomain.CONDUIT, "flow_dir": FlowDirection.IN,
             "system": SystemType.LIGHTING, "shape": ConnectorShape.ROUND,
             "size_param": "Conduit_Size", "pos_offset": (0, 0, 1), "dir": (0, 0, 1),
             "voltage": 230, "poles": 1},
        ],
        param_specs=[
            {"name": "Wattage", "type": ParameterType.POWER, "group": ParameterGroup.ELECTRICAL,
             "unit": "W", "range": (5, 150)},
            {"name": "Luminous_Flux", "type": ParameterType.NUMBER, "group": ParameterGroup.ELECTRICAL,
             "unit": "lm", "range": (300, 15000)},
            {"name": "Color_Temperature", "type": ParameterType.NUMBER, "group": ParameterGroup.ELECTRICAL,
             "unit": "K", "range": (2700, 6500)},
            {"name": "CRI", "type": ParameterType.NUMBER, "group": ParameterGroup.ELECTRICAL,
             "unit": "", "range": (80, 98)},
            {"name": "Efficacy", "type": ParameterType.NUMBER, "group": ParameterGroup.ELECTRICAL,
             "unit": "lm/W", "range": (80, 170)},
            {"name": "Dimming_Min", "type": ParameterType.NUMBER, "group": ParameterGroup.ELECTRICAL,
             "unit": "%", "range": (1, 10)},
        ],
        formula_specs=[
            {"target": "Efficacy", "expr": "Luminous_Flux / Wattage",
             "deps": ["Luminous_Flux", "Wattage"], "desc": "Световая отдача"},
        ],
        type_generator="light_fixture",
    ),
    # Switch
    EquipmentTemplate(
        name_pattern="BIM_AI_Switch_{type}",
        category="Lighting Devices",
        mep_domain=MEPDomain.ELECTRICAL,
        subcategory="Switches",
        host_type="Wall",
        ifc_class="IfcSwitchingDevice",
        ifc_predefined_type="TOGGLESWITCH",
        omniclass="23-35 17 15",
        template_file="Metric Electrical Fixture - Wall Hosted.rft",
        tags=["electrical", "switch", "lighting_control"],
        connector_specs=[
            {"domain": ConnectorDomain.CONDUIT, "flow_dir": FlowDirection.BIDIRECTIONAL,
             "system": SystemType.LIGHTING, "shape": ConnectorShape.ROUND,
             "size_param": "Conduit_Size", "pos_offset": (0, 0, -1), "dir": (0, 0, -1),
             "voltage": 230, "poles": 1},
        ],
        param_specs=[
            {"name": "Voltage", "type": ParameterType.VOLTAGE, "group": ParameterGroup.ELECTRICAL,
             "unit": "V", "range": (220, 230)},
            {"name": "Amperage", "type": ParameterType.CURRENT, "group": ParameterGroup.ELECTRICAL,
             "unit": "A", "range": (10, 16)},
            {"name": "Number_of_Gangs", "type": ParameterType.INTEGER, "group": ParameterGroup.ELECTRICAL,
             "unit": "", "range": (1, 4)},
            {"name": "Mounting_Height", "type": ParameterType.LENGTH, "group": ParameterGroup.DIMENSIONS,
             "unit": "mm", "range": (900, 1200)},
        ],
        type_generator="switch",
    ),
]


# ─────────────────────────────────────────────
# Fire Protection Equipment Templates
# ─────────────────────────────────────────────

FIRE_PROTECTION_TEMPLATES = [
    # Sprinkler Head
    EquipmentTemplate(
        name_pattern="BIM_AI_Sprinkler_{type}_{kfactor}",
        category="Sprinklers",
        mep_domain=MEPDomain.FIRE_PROTECTION,
        subcategory="Pendant Sprinklers",
        host_type="Ceiling",
        ifc_class="IfcFireSuppressionTerminal",
        ifc_predefined_type="SPRINKLER",
        omniclass="23-37 11 11",
        template_file="Metric Sprinkler.rft",
        tags=["fire_protection", "sprinkler", "pendant", "wet"],
        connector_specs=[
            {"domain": ConnectorDomain.PIPE, "flow_dir": FlowDirection.IN,
             "system": SystemType.WET_SPRINKLER, "shape": ConnectorShape.ROUND,
             "size_param": "Connection_Size", "pos_offset": (0, 0, 1), "dir": (0, 0, 1)},
        ],
        param_specs=[
            {"name": "K_Factor", "type": ParameterType.NUMBER, "group": ParameterGroup.FIRE_PROTECTION,
             "unit": "L/min/bar^0.5", "range": (57, 363)},
            {"name": "Activation_Temperature", "type": ParameterType.TEMPERATURE,
             "group": ParameterGroup.FIRE_PROTECTION,
             "unit": "°C", "range": (57, 141)},
            {"name": "Coverage_Area", "type": ParameterType.AREA, "group": ParameterGroup.FIRE_PROTECTION,
             "unit": "m²", "range": (6, 21)},
            {"name": "Response_Type", "type": ParameterType.TEXT, "group": ParameterGroup.FIRE_PROTECTION,
             "unit": "", "range": None},
            {"name": "Orifice_Size", "type": ParameterType.LENGTH, "group": ParameterGroup.DIMENSIONS,
             "unit": "mm", "range": (10, 25)},
        ],
        type_generator="sprinkler",
    ),
    # Smoke Detector
    EquipmentTemplate(
        name_pattern="BIM_AI_SmokeDetector_{type}",
        category="Fire Alarm Devices",
        mep_domain=MEPDomain.FIRE_PROTECTION,
        subcategory="Smoke Detectors",
        host_type="Ceiling",
        ifc_class="IfcSensor",
        ifc_predefined_type="SMOKESENSOR",
        omniclass="23-37 15 11",
        template_file="Metric Fire Alarm Device - Ceiling Hosted.rft",
        tags=["fire_protection", "fire_alarm", "smoke_detector", "sensor"],
        connector_specs=[
            {"domain": ConnectorDomain.CONDUIT, "flow_dir": FlowDirection.BIDIRECTIONAL,
             "system": SystemType.FIRE_ALARM, "shape": ConnectorShape.ROUND,
             "size_param": "Cable_Size", "pos_offset": (0, 0, 1), "dir": (0, 0, 1),
             "voltage": 24, "poles": 2},
        ],
        param_specs=[
            {"name": "Coverage_Area", "type": ParameterType.AREA, "group": ParameterGroup.FIRE_PROTECTION,
             "unit": "m²", "range": (40, 100)},
            {"name": "Sensitivity", "type": ParameterType.NUMBER, "group": ParameterGroup.FIRE_PROTECTION,
             "unit": "%/m", "range": (0.5, 12)},
            {"name": "Voltage", "type": ParameterType.VOLTAGE, "group": ParameterGroup.ELECTRICAL,
             "unit": "V DC", "range": (18, 32)},
            {"name": "Max_Ceiling_Height", "type": ParameterType.LENGTH, "group": ParameterGroup.FIRE_PROTECTION,
             "unit": "m", "range": (3, 12)},
        ],
        type_generator="smoke_detector",
    ),
    # Fire Damper
    EquipmentTemplate(
        name_pattern="BIM_AI_FireDamper_{shape}_{size}",
        category="Duct Accessories",
        mep_domain=MEPDomain.FIRE_PROTECTION,
        subcategory="Fire Dampers",
        host_type="Wall",
        ifc_class="IfcDamper",
        ifc_predefined_type="FIREDAMPER",
        omniclass="23-37 19 11",
        template_file="Metric Duct Accessory.rft",
        tags=["fire_protection", "fire_damper", "duct_accessory", "rated"],
        connector_specs=[
            {"domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.BIDIRECTIONAL,
             "system": SystemType.SUPPLY_AIR, "shape": ConnectorShape.RECTANGULAR,
             "size_param": "Duct_Width", "pos_offset": (-1, 0, 0), "dir": (-1, 0, 0)},
            {"domain": ConnectorDomain.DUCT, "flow_dir": FlowDirection.BIDIRECTIONAL,
             "system": SystemType.SUPPLY_AIR, "shape": ConnectorShape.RECTANGULAR,
             "size_param": "Duct_Width", "pos_offset": (1, 0, 0), "dir": (1, 0, 0)},
        ],
        param_specs=[
            {"name": "Fire_Rating", "type": ParameterType.TEXT, "group": ParameterGroup.FIRE_PROTECTION,
             "unit": "", "range": None},
            {"name": "Leakage_Class", "type": ParameterType.TEXT, "group": ParameterGroup.FIRE_PROTECTION,
             "unit": "", "range": None},
            {"name": "Pressure_Drop", "type": ParameterType.PRESSURE, "group": ParameterGroup.MECHANICAL,
             "unit": "Pa", "range": (3, 15)},
            {"name": "Actuator_Type", "type": ParameterType.TEXT, "group": ParameterGroup.MECHANICAL,
             "unit": "", "range": None},
        ],
        type_generator="fire_damper",
    ),
]


ALL_TEMPLATES = HVAC_TEMPLATES + PIPING_TEMPLATES + ELECTRICAL_TEMPLATES + FIRE_PROTECTION_TEMPLATES


# ─────────────────────────────────────────────
# Type Variant Generators
# ─────────────────────────────────────────────

def _pick_dn(min_dn: int = 15, max_dn: int = 150) -> tuple[int, str]:
    """Pick a random pipe DN in valid range."""
    valid = [d for d in PIPE_DN if min_dn <= d <= max_dn]
    dn = random.choice(valid)
    return dn, PIPE_DN_LABELS[dn]


def gen_types_diffuser_round(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for diam in random.sample(DUCT_ROUND_DIAMETERS[:10], min(4, len(DUCT_ROUND_DIAMETERS[:10]))):
        airflow = round(math.pi * (diam / 2000) ** 2 * random.uniform(3, 6) * 1000, 0)
        types.append(FamilyTypeVariant(
            name=f"Ø{diam} - {int(airflow)} L/s",
            parameter_values={
                "Duct_Diameter": diam,
                "Face_Diameter": diam + 100,
                "Airflow": airflow,
                "Throw": round(airflow / 60, 1),
                "Noise_Level": random.randint(18, 35),
            },
        ))
    return types


def gen_types_grille_rect(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for w in random.sample(DUCT_RECT_WIDTHS[:6], min(3, 6)):
        h = random.choice([h for h in DUCT_RECT_HEIGHTS if h <= w])
        airflow = round(w * h / 1e6 * random.uniform(3, 5) * 1000, 0)
        types.append(FamilyTypeVariant(
            name=f"{w}x{h}mm - {int(airflow)} L/s",
            parameter_values={
                "Duct_Width": w, "Duct_Height": h,
                "Airflow": airflow,
                "Effective_Area": round(w * h / 1e6 * 0.75, 3),
            },
        ))
    return types


def gen_types_fcu_4pipe(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for cooling_kw in [2, 3.5, 5, 7, 10]:
        airflow = int(cooling_kw * 80)
        duct_d = random.choice([d for d in DUCT_ROUND_DIAMETERS if d >= airflow * 0.3 and d <= airflow * 0.8])
        pipe_dn_val, pipe_dn_lbl = _pick_dn(20, 40)
        types.append(FamilyTypeVariant(
            name=f"FCU-4P-{cooling_kw}kW",
            parameter_values={
                "Cooling_Capacity": cooling_kw,
                "Heating_Capacity": round(cooling_kw * 1.2, 1),
                "Airflow": airflow,
                "Supply_Duct_Diameter": duct_d,
                "Return_Duct_Diameter": duct_d,
                "CHW_Pipe_Size": pipe_dn_val,
                "CHW_Return_Pipe_Size": pipe_dn_val,
                "HW_Pipe_Size": pipe_dn_val,
                "HW_Return_Pipe_Size": pipe_dn_val,
                "Fan_Power": int(airflow * 0.4),
            },
        ))
    return types


def gen_types_vav_box(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for diam in [150, 200, 250, 300, 400]:
        max_flow = int(math.pi * (diam / 2000) ** 2 * 8 * 1000)
        types.append(FamilyTypeVariant(
            name=f"VAV-Ø{diam}",
            parameter_values={
                "Inlet_Diameter": diam, "Outlet_Diameter": diam,
                "Max_Airflow": max_flow,
                "Min_Airflow": int(max_flow * 0.3),
                "Pressure_Drop": random.randint(30, 80),
            },
        ))
    return types


def gen_types_ahu(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for flow in [2000, 5000, 10000, 20000, 40000]:
        types.append(FamilyTypeVariant(
            name=f"AHU-{flow}m³h",
            parameter_values={
                "Supply_Airflow": flow,
                "ESP": random.choice([300, 500, 800, 1000]),
                "Cooling_Capacity": round(flow * 0.012, 0),
                "Heating_Capacity": round(flow * 0.008, 0),
                "Fan_Power": round(flow * 0.0003, 1),
                "Heat_Recovery_Efficiency": random.choice([70, 75, 80, 85]),
                "Filter_Class": random.choice(["F7", "F9", "H13"]),
            },
        ))
    return types


def gen_types_washbasin(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    return [
        FamilyTypeVariant(name="Standard 500x400", parameter_values={
            "Basin_Width": 500, "Basin_Depth": 400, "Mounting_Height": 850,
            "CW_Connection": 15, "HW_Connection": 15, "Drain_Connection": 40,
            "Fixture_Units": 1, "Flow_Rate": 0.15,
        }),
        FamilyTypeVariant(name="Compact 400x300", parameter_values={
            "Basin_Width": 400, "Basin_Depth": 300, "Mounting_Height": 850,
            "CW_Connection": 15, "HW_Connection": 15, "Drain_Connection": 32,
            "Fixture_Units": 0.5, "Flow_Rate": 0.1,
        }),
        FamilyTypeVariant(name="Accessible 600x500", parameter_values={
            "Basin_Width": 600, "Basin_Depth": 500, "Mounting_Height": 800,
            "CW_Connection": 15, "HW_Connection": 15, "Drain_Connection": 40,
            "Fixture_Units": 1.5, "Flow_Rate": 0.2,
        }),
    ]


def gen_types_pump(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for flow, head in [(1, 4), (3, 6), (8, 10), (20, 15), (50, 20)]:
        dn_val, _ = _pick_dn(25, 150)
        power = round((flow * head * 9.81) / (3600 * 0.7), 2)
        types.append(FamilyTypeVariant(
            name=f"Pump-{flow}m³h-{head}m",
            parameter_values={
                "Flow_Rate": flow, "Head": head, "Motor_Power": power,
                "Inlet_DN": dn_val, "Outlet_DN": dn_val,
                "RPM": random.choice([1450, 2900]),
            },
        ))
    return types


def gen_types_water_heater(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for vol in [50, 100, 200, 500]:
        power = round(vol * 0.04, 1)
        dn_val, _ = _pick_dn(20, 50)
        types.append(FamilyTypeVariant(
            name=f"WH-{vol}L",
            parameter_values={
                "Volume": vol, "Heating_Power": power,
                "Max_Temperature": 65,
                "CW_Inlet_DN": dn_val, "HW_Outlet_DN": dn_val,
                "Recirc_DN": min(dn_val, 25),
            },
        ))
    return types


def gen_types_panel(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for amps in [63, 100, 160, 250, 400]:
        ways = random.choice([12, 24, 36, 48, 72])
        types.append(FamilyTypeVariant(
            name=f"Panel-{amps}A-{ways}way",
            parameter_values={
                "Main_Bus_Rating": amps, "Number_of_Ways": ways,
                "Voltage": 400, "Short_Circuit_Rating": random.choice([10, 25, 36, 50]),
                "Total_Connected_Load": round(amps * 400 * 0.8 * 1.73 / 1000, 1),
            },
        ))
    return types


def gen_types_outlet(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    return [
        FamilyTypeVariant(name="Single-16A", parameter_values={
            "Voltage": 230, "Amperage": 16, "Number_of_Gangs": 1,
            "Mounting_Height": 300, "Conduit_Size": 20,
        }),
        FamilyTypeVariant(name="Double-16A", parameter_values={
            "Voltage": 230, "Amperage": 16, "Number_of_Gangs": 2,
            "Mounting_Height": 300, "Conduit_Size": 20,
        }),
        FamilyTypeVariant(name="Industrial-32A", parameter_values={
            "Voltage": 400, "Amperage": 32, "Number_of_Gangs": 1,
            "Mounting_Height": 500, "Conduit_Size": 32,
        }),
    ]


def gen_types_light_fixture(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for watt, lumens in [(18, 2000), (36, 4200), (45, 5400), (72, 9000), (120, 14000)]:
        types.append(FamilyTypeVariant(
            name=f"LED-{watt}W-{lumens}lm",
            parameter_values={
                "Wattage": watt, "Luminous_Flux": lumens,
                "Color_Temperature": random.choice([3000, 4000, 5000]),
                "CRI": random.choice([80, 90, 95]),
                "Efficacy": round(lumens / watt, 1),
                "Dimming_Min": random.choice([1, 5, 10]),
                "Conduit_Size": 20,
            },
        ))
    return types


def gen_types_switch(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    return [
        FamilyTypeVariant(name="Single", parameter_values={
            "Voltage": 230, "Amperage": 10, "Number_of_Gangs": 1, "Mounting_Height": 1100, "Conduit_Size": 20,
        }),
        FamilyTypeVariant(name="Double", parameter_values={
            "Voltage": 230, "Amperage": 10, "Number_of_Gangs": 2, "Mounting_Height": 1100, "Conduit_Size": 20,
        }),
        FamilyTypeVariant(name="Dimmer", parameter_values={
            "Voltage": 230, "Amperage": 3, "Number_of_Gangs": 1, "Mounting_Height": 1100, "Conduit_Size": 20,
        }),
    ]


def gen_types_sprinkler(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for k in SPRINKLER_K_FACTORS:
        temp = random.choice([57, 68, 79, 93, 141])
        types.append(FamilyTypeVariant(
            name=f"K{k}-{temp}°C",
            parameter_values={
                "K_Factor": k,
                "Activation_Temperature": temp,
                "Coverage_Area": random.choice([9, 12, 16, 21]),
                "Response_Type": random.choice(["Standard", "Quick Response", "ESFR"]),
                "Connection_Size": random.choice([15, 20]),
                "Orifice_Size": round(math.sqrt(k / 14) * 2, 1),
            },
        ))
    return types


def gen_types_smoke_detector(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    return [
        FamilyTypeVariant(name="Optical", parameter_values={
            "Coverage_Area": 60, "Sensitivity": 3.0, "Voltage": 24,
            "Max_Ceiling_Height": 8, "Cable_Size": 1.5,
        }),
        FamilyTypeVariant(name="Thermal", parameter_values={
            "Coverage_Area": 50, "Sensitivity": 0, "Voltage": 24,
            "Max_Ceiling_Height": 6, "Cable_Size": 1.5,
        }),
        FamilyTypeVariant(name="Multi-Sensor", parameter_values={
            "Coverage_Area": 80, "Sensitivity": 2.0, "Voltage": 24,
            "Max_Ceiling_Height": 12, "Cable_Size": 1.5,
        }),
    ]


def gen_types_fire_damper(template: EquipmentTemplate) -> list[FamilyTypeVariant]:
    types = []
    for w in [200, 300, 400, 600, 800]:
        for h in [200, 300, 400]:
            if h > w:
                continue
            types.append(FamilyTypeVariant(
                name=f"FD-{w}x{h}",
                parameter_values={
                    "Duct_Width": w, "Duct_Height": h,
                    "Fire_Rating": random.choice(["EI60", "EI90", "EI120"]),
                    "Leakage_Class": random.choice(["C", "D"]),
                    "Actuator_Type": random.choice(["Spring Return", "Motorized", "Fusible Link"]),
                    "Pressure_Drop": random.randint(3, 12),
                },
            ))
    return random.sample(types, min(5, len(types)))


TYPE_GENERATORS: dict[str, Callable] = {
    "diffuser_round": gen_types_diffuser_round,
    "grille_rect": gen_types_grille_rect,
    "fcu_4pipe": gen_types_fcu_4pipe,
    "vav_box": gen_types_vav_box,
    "ahu": gen_types_ahu,
    "washbasin": gen_types_washbasin,
    "pump": gen_types_pump,
    "water_heater": gen_types_water_heater,
    "panel": gen_types_panel,
    "outlet": gen_types_outlet,
    "light_fixture": gen_types_light_fixture,
    "switch": gen_types_switch,
    "sprinkler": gen_types_sprinkler,
    "smoke_detector": gen_types_smoke_detector,
    "fire_damper": gen_types_fire_damper,
}


# ─────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────

class MEPFamilyGenerator:
    """Generates synthetic MEP families from templates."""

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

    def _build_connector(self, spec: dict, family_size_hint: float = 300) -> MEPConnector:
        """Build MEPConnector from template spec."""
        shape = spec.get("shape", ConnectorShape.ROUND)
        dim = ConnectorDimensions()
        if shape == ConnectorShape.ROUND:
            dim.diameter = random.choice(DUCT_ROUND_DIAMETERS[:8])
        elif shape == ConnectorShape.RECTANGULAR:
            dim.width = random.choice(DUCT_RECT_WIDTHS[:6])
            dim.height = random.choice(DUCT_RECT_HEIGHTS[:4])

        if spec["domain"] == ConnectorDomain.PIPE:
            dn_val, dn_lbl = _pick_dn()
            dim = ConnectorDimensions(nominal_diameter=dn_lbl, diameter=dn_val)

        if spec["domain"] == ConnectorDomain.CONDUIT:
            dim = ConnectorDimensions(diameter=random.choice(CONDUIT_DIAMETERS[:4]))

        pos = tuple(p * family_size_hint / 2 for p in spec["pos_offset"])
        return MEPConnector(
            id=f"conn_{spec['system'].value.lower()}_{uuid.uuid4().hex[:4]}",
            name=f"{spec['system'].value} Connection",
            domain=spec["domain"],
            flow_direction=spec["flow_dir"],
            system_type=spec["system"],
            shape=shape,
            dimensions=dim,
            position=pos,
            direction=spec["dir"],
            linked_size_param=spec.get("size_param"),
            voltage=spec.get("voltage"),
            number_of_poles=spec.get("poles"),
        )

    def _build_calc_param(self, spec: dict) -> CalculationParameter:
        """Build CalculationParameter from template spec."""
        default_val = None
        if spec["range"] is not None:
            lo, hi = spec["range"]
            if spec["type"] == ParameterType.INTEGER:
                default_val = random.randint(int(lo), int(hi))
            else:
                default_val = round(random.uniform(lo, hi), 2)

        return CalculationParameter(
            name=spec["name"],
            display_name=spec["name"].replace("_", " "),
            param_type=spec["type"],
            group=spec["group"],
            unit_label=spec["unit"],
            default_value=default_val,
            min_value=spec["range"][0] if spec["range"] else None,
            max_value=spec["range"][1] if spec["range"] else None,
        )

    def _build_formula(self, spec: dict) -> FormulaDefinition:
        return FormulaDefinition(
            target_param=spec["target"],
            expression=spec["expr"],
            description=spec.get("desc", ""),
            dependencies=spec.get("deps", []),
        )

    def generate_family(self, template: EquipmentTemplate) -> MEPBIMFamily:
        """Generate a complete MEP family from a template."""
        manufacturer = random.choice(MANUFACTURERS.get(template.mep_domain.value, ["Generic"]))
        model_number = f"{random.choice('ABCDEFG')}{random.randint(100,999)}"

        connectors = [self._build_connector(spec) for spec in template.connector_specs]
        params = [self._build_calc_param(spec) for spec in template.param_specs]
        formulas = [self._build_formula(spec) for spec in template.formula_specs]

        # Apply formulas to params
        for f in formulas:
            for p in params:
                if p.name == f.target_param:
                    p.formula = f

        # Generate type variants
        type_gen = TYPE_GENERATORS.get(template.type_generator)
        family_types = type_gen(template) if type_gen else []

        # System requirements
        sys_req = SystemRequirements(
            min_connectors=len(template.connector_specs),
            max_connectors=len(template.connector_specs) + 2,
            required_connector_domains=list({s["domain"] for s in template.connector_specs}),
            required_system_types=list({s["system"] for s in template.connector_specs}),
            requires_electrical=any(s["domain"] == ConnectorDomain.CONDUIT for s in template.connector_specs),
            requires_flow_params=any(s["domain"] in (ConnectorDomain.DUCT, ConnectorDomain.PIPE) for s in template.connector_specs),
            allowed_hosting=[template.host_type],
        )

        # All possible name slots — safe formatting
        name_slots = dict(
            size=random.randint(100, 500),
            type=random.choice(["A", "B", "C"]),
            config="4P",
            capacity=random.randint(2, 20),
            airflow=f"{random.randint(1, 50)}k",
            flow=f"{random.randint(1, 100)}m3h",
            volume=random.choice([50, 100, 200, 500]),
            amps=random.choice(AMPERE_RATINGS),
            wattage=random.choice([18, 36, 45, 72]),
            kfactor=random.choice(SPRINKLER_K_FACTORS),
            shape=random.choice(["Round", "Rect"]),
        )
        try:
            family_name = template.name_pattern.format(**name_slots)
        except KeyError:
            family_name = template.name_pattern.split("_{")[0] + f"_{random.randint(100,999)}"

        family = MEPBIMFamily(
            family_name=family_name,
            category=template.category,
            mep_domain=template.mep_domain,
            subcategory=template.subcategory,
            description=f"{template.subcategory} by {manufacturer} ({model_number})",
            manufacturer=manufacturer,
            model=model_number,
            classification=ClassificationReference(
                omniclass_table=template.omniclass,
                ifc_class=template.ifc_class,
                ifc_predefined_type=template.ifc_predefined_type,
            ),
            host_type=template.host_type,
            connectors=connectors,
            calculation_params=params,
            formulas=formulas,
            geometry=[GeometryPrimitive(type=g) for g in template.geometry_types],
            family_types=family_types,
            system_requirements=sys_req,
            template_file=template.template_file,
            tags=template.tags,
        )
        return family

    def generate_dataset(
        self,
        n_per_template: int = 3,
        templates: list[EquipmentTemplate] | None = None,
    ) -> list[dict]:
        """Generate a full dataset of MEP families."""
        templates = templates or ALL_TEMPLATES
        dataset = []
        for tmpl in templates:
            for i in range(n_per_template):
                family = self.generate_family(tmpl)
                dataset.append(family.to_dict())
        return dataset

    def generate_by_domain(self, domain: MEPDomain, n_per_template: int = 3) -> list[dict]:
        """Generate families for a specific MEP domain."""
        templates = [t for t in ALL_TEMPLATES if t.mep_domain == domain]
        return self.generate_dataset(n_per_template, templates)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    gen = MEPFamilyGenerator(seed=42)

    if "--all" in sys.argv:
        dataset = gen.generate_dataset(n_per_template=3)
    elif "--domain" in sys.argv:
        idx = sys.argv.index("--domain") + 1
        domain = MEPDomain(sys.argv[idx])
        dataset = gen.generate_dataset(n_per_template=3, templates=[
            t for t in ALL_TEMPLATES if t.mep_domain == domain
        ])
    else:
        # Quick demo — one of each
        dataset = gen.generate_dataset(n_per_template=1)

    print(json.dumps(dataset, indent=2, ensure_ascii=False))
    print(f"\n✅ Generated {len(dataset)} MEP families across {len(set(d['category'] for d in dataset))} categories")
    print(f"   Domains: {sorted(set(d['mep_domain'] for d in dataset))}")
    print(f"   Total connectors: {sum(len(d['connectors']) for d in dataset)}")
    print(f"   Total type variants: {sum(len(d['family_types']) for d in dataset)}")
