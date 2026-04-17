"""
BIM.AI — MEP Constraint Validator
====================================
Валидирует сгенерированные MEP-семейства на:
1. Корректность коннекторов (наличие, позиции, направления, домены)
2. Допустимые диаметры/размеры (стандартные ряды)
3. Совместимость систем (нельзя подключить pipe к duct)
4. Расчётные параметры (диапазоны, формулы, зависимости)
5. Классификация (IFC / OmniClass)
6. Общая структурная целостность

Каждая проверка возвращает ValidationResult с severity:
  ERROR   — семейство невалидно, нельзя использовать
  WARNING — проблема, но семейство может работать
  INFO    — рекомендация по улучшению
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math
import json


# ─────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────

class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationResult:
    rule: str
    severity: Severity
    message: str
    field_path: str = ""
    suggestion: str = ""


@dataclass
class ValidationReport:
    family_name: str = ""
    results: list[ValidationResult] = field(default_factory=list)
    is_valid: bool = True

    @property
    def errors(self) -> list[ValidationResult]:
        return [r for r in self.results if r.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationResult]:
        return [r for r in self.results if r.severity == Severity.WARNING]

    def add(self, result: ValidationResult):
        self.results.append(result)
        if result.severity == Severity.ERROR:
            self.is_valid = False

    def summary(self) -> str:
        lines = [f"Validation: {self.family_name}"]
        lines.append(f"  Valid: {'✅' if self.is_valid else '❌'}")
        lines.append(f"  Errors: {len(self.errors)}, Warnings: {len(self.warnings)}, "
                     f"Info: {len(self.results) - len(self.errors) - len(self.warnings)}")
        for r in self.results:
            icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}[r.severity.value]
            lines.append(f"  {icon} [{r.rule}] {r.message}")
            if r.suggestion:
                lines.append(f"      → {r.suggestion}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Standard Engineering Constants
# ─────────────────────────────────────────────

VALID_DUCT_ROUND_DIAMETERS = {
    80, 100, 112, 125, 140, 150, 160, 180, 200, 224, 250, 280, 300,
    315, 355, 400, 450, 500, 560, 630, 710, 800, 900, 1000, 1120, 1250,
}

VALID_PIPE_DN = {
    6, 8, 10, 15, 20, 25, 32, 40, 50, 65, 80, 100, 125, 150, 200,
    250, 300, 350, 400, 450, 500, 600,
}

VALID_CONDUIT_DIAMETERS = {16, 20, 25, 32, 40, 50, 63}

# Domain → allowed system types
DOMAIN_SYSTEM_COMPATIBILITY = {
    "DomainHvac": {
        "SupplyAir", "ReturnAir", "ExhaustAir", "OutsideAir",
    },
    "DomainPiping": {
        "DomesticHotWater", "DomesticColdWater", "Sanitary", "Vent",
        "HydronicSupply", "HydronicReturn", "ChilledWater",
        "CondenserWater", "Refrigerant",
        "WetSprinkler", "DrySprinkler", "PreAction",
    },
    "DomainElectrical": {
        "Power", "Lighting", "FireAlarm", "Data", "Communication",
    },
    "DomainCableTray": {
        "Power", "Data", "Communication",
    },
}

# Revit category → allowed connector domains
CATEGORY_DOMAIN_MAP = {
    "Air Terminals": {"DomainHvac"},
    "Duct Accessories": {"DomainHvac"},
    "Duct Fittings": {"DomainHvac"},
    "Flex Ducts": {"DomainHvac"},
    "Mechanical Equipment": {"DomainHvac", "DomainPiping", "DomainElectrical"},
    "Plumbing Fixtures": {"DomainPiping", "DomainElectrical"},
    "Plumbing Equipment": {"DomainPiping", "DomainElectrical"},
    "Pipe Accessories": {"DomainPiping"},
    "Pipe Fittings": {"DomainPiping"},
    "Electrical Equipment": {"DomainElectrical", "DomainCableTray"},
    "Electrical Fixtures": {"DomainElectrical"},
    "Lighting Fixtures": {"DomainElectrical"},
    "Lighting Devices": {"DomainElectrical"},
    "Communication Devices": {"DomainElectrical"},
    "Data Devices": {"DomainElectrical"},
    "Fire Alarm Devices": {"DomainElectrical"},
    "Sprinklers": {"DomainPiping"},
    "Fire Protection Equipment": {"DomainPiping", "DomainElectrical"},
    "Cable Tray Fittings": {"DomainCableTray"},
    "Conduit Fittings": {"DomainElectrical"},
}

# IFC class → expected Revit categories
IFC_CATEGORY_MAP = {
    "IfcAirTerminal": {"Air Terminals"},
    "IfcFan": {"Mechanical Equipment"},
    "IfcUnitaryEquipment": {"Mechanical Equipment"},
    "IfcAirToAirHeatRecovery": {"Mechanical Equipment"},
    "IfcDamper": {"Duct Accessories"},
    "IfcSanitaryTerminal": {"Plumbing Fixtures"},
    "IfcPump": {"Plumbing Equipment"},
    "IfcBoiler": {"Plumbing Equipment", "Mechanical Equipment"},
    "IfcChiller": {"Mechanical Equipment"},
    "IfcElectricDistributionBoard": {"Electrical Equipment"},
    "IfcOutlet": {"Electrical Fixtures"},
    "IfcLightFixture": {"Lighting Fixtures"},
    "IfcSwitchingDevice": {"Lighting Devices", "Electrical Fixtures"},
    "IfcFireSuppressionTerminal": {"Sprinklers"},
    "IfcSensor": {"Fire Alarm Devices", "Communication Devices"},
}

# Host type → valid Revit template suffixes
HOST_TEMPLATE_MAP = {
    "Wall": ["Wall Hosted", "Wall Based"],
    "Ceiling": ["Ceiling Hosted", "Ceiling Based"],
    "Floor": ["Floor Hosted", "Floor Based"],
    "Face": ["Face Based", "Face Hosted"],
    "Non-hosted": [""],
}

# Physical limits for sanity checks
PARAM_PHYSICAL_LIMITS = {
    "Flow": {"min": 0, "max": 200000, "unit": "L/s"},         # 0 — 200 m³/s
    "Pressure": {"min": -5000, "max": 100000, "unit": "Pa"},
    "Power": {"min": 0, "max": 10000000, "unit": "W"},        # 0 — 10 MW
    "Voltage": {"min": 0, "max": 1000, "unit": "V"},
    "Current": {"min": 0, "max": 6300, "unit": "A"},
    "Temperature": {"min": -50, "max": 500, "unit": "°C"},
    "Length": {"min": 0, "max": 50000, "unit": "mm"},
    "Area": {"min": 0, "max": 10000, "unit": "m²"},
    "Volume": {"min": 0, "max": 100000, "unit": "L"},
}


# ─────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────

class MEPValidator:
    """Validates MEP family JSON against engineering rules."""

    def validate(self, family: dict) -> ValidationReport:
        report = ValidationReport(family_name=family.get("family_name", "<unknown>"))

        self._check_required_fields(family, report)
        self._check_connectors(family, report)
        self._check_connector_system_compatibility(family, report)
        self._check_connector_sizes(family, report)
        self._check_connector_positions(family, report)
        self._check_flow_balance(family, report)
        self._check_parameters(family, report)
        self._check_formulas(family, report)
        self._check_classification(family, report)
        self._check_host_type(family, report)
        self._check_type_variants(family, report)
        self._check_category_domain(family, report)

        return report

    # ── Required fields ──────────────────────

    def _check_required_fields(self, f: dict, r: ValidationReport):
        for field in ["family_name", "category", "mep_domain", "connectors"]:
            if not f.get(field):
                r.add(ValidationResult(
                    rule="REQUIRED_FIELD",
                    severity=Severity.ERROR,
                    message=f"Missing required field: {field}",
                    field_path=field,
                ))

        if f.get("family_name") and len(f["family_name"]) > 200:
            r.add(ValidationResult(
                rule="NAME_LENGTH",
                severity=Severity.WARNING,
                message="Family name exceeds 200 characters",
                field_path="family_name",
                suggestion="Shorten the name for Revit compatibility",
            ))

    # ── Connector validation ─────────────────

    def _check_connectors(self, f: dict, r: ValidationReport):
        connectors = f.get("connectors", [])
        if not connectors:
            r.add(ValidationResult(
                rule="MIN_CONNECTORS",
                severity=Severity.ERROR,
                message="MEP family must have at least 1 connector",
                field_path="connectors",
            ))
            return

        if len(connectors) > 30:
            r.add(ValidationResult(
                rule="MAX_CONNECTORS",
                severity=Severity.WARNING,
                message=f"Family has {len(connectors)} connectors — excessive for most equipment",
                field_path="connectors",
            ))

        ids = [c.get("id", "") for c in connectors]
        if len(ids) != len(set(ids)):
            r.add(ValidationResult(
                rule="DUPLICATE_CONNECTOR_ID",
                severity=Severity.ERROR,
                message="Connector IDs must be unique",
                field_path="connectors",
            ))

        for i, conn in enumerate(connectors):
            if not conn.get("domain"):
                r.add(ValidationResult(
                    rule="CONNECTOR_DOMAIN",
                    severity=Severity.ERROR,
                    message=f"Connector [{i}] missing domain",
                    field_path=f"connectors[{i}].domain",
                ))

            if not conn.get("flow_direction"):
                r.add(ValidationResult(
                    rule="CONNECTOR_FLOW_DIR",
                    severity=Severity.ERROR,
                    message=f"Connector [{i}] missing flow_direction",
                    field_path=f"connectors[{i}].flow_direction",
                ))

    def _check_connector_system_compatibility(self, f: dict, r: ValidationReport):
        for i, conn in enumerate(f.get("connectors", [])):
            domain = conn.get("domain", "")
            system = conn.get("system_type", "")

            if domain in DOMAIN_SYSTEM_COMPATIBILITY:
                allowed = DOMAIN_SYSTEM_COMPATIBILITY[domain]
                if system and system not in allowed:
                    r.add(ValidationResult(
                        rule="SYSTEM_DOMAIN_MISMATCH",
                        severity=Severity.ERROR,
                        message=(f"Connector [{i}]: system '{system}' incompatible with "
                                f"domain '{domain}'"),
                        field_path=f"connectors[{i}]",
                        suggestion=f"Allowed systems for {domain}: {sorted(allowed)}",
                    ))

    def _check_connector_sizes(self, f: dict, r: ValidationReport):
        for i, conn in enumerate(f.get("connectors", [])):
            dims = conn.get("dimensions", {})
            domain = conn.get("domain", "")
            shape = conn.get("shape", "")

            if domain == "DomainHvac" and shape == "Round":
                diam = dims.get("diameter")
                if diam and diam not in VALID_DUCT_ROUND_DIAMETERS:
                    # Find nearest standard size
                    nearest = min(VALID_DUCT_ROUND_DIAMETERS, key=lambda d: abs(d - diam))
                    r.add(ValidationResult(
                        rule="NON_STANDARD_DUCT_SIZE",
                        severity=Severity.WARNING,
                        message=f"Connector [{i}]: duct Ø{diam}mm not in standard range",
                        field_path=f"connectors[{i}].dimensions.diameter",
                        suggestion=f"Nearest standard: Ø{nearest}mm",
                    ))

            if domain == "DomainPiping":
                diam = dims.get("diameter")
                nom = dims.get("nominal_diameter", "")
                check_dn = diam
                if nom and nom.startswith("DN"):
                    try:
                        check_dn = int(nom[2:])
                    except ValueError:
                        pass
                if check_dn and check_dn not in VALID_PIPE_DN:
                    nearest = min(VALID_PIPE_DN, key=lambda d: abs(d - check_dn))
                    r.add(ValidationResult(
                        rule="NON_STANDARD_PIPE_SIZE",
                        severity=Severity.WARNING,
                        message=f"Connector [{i}]: pipe DN{check_dn} not standard",
                        field_path=f"connectors[{i}].dimensions",
                        suggestion=f"Nearest standard: DN{nearest}",
                    ))

            if domain == "DomainElectrical":
                diam = dims.get("diameter")
                if diam and diam not in VALID_CONDUIT_DIAMETERS:
                    nearest = min(VALID_CONDUIT_DIAMETERS, key=lambda d: abs(d - diam))
                    r.add(ValidationResult(
                        rule="NON_STANDARD_CONDUIT_SIZE",
                        severity=Severity.WARNING,
                        message=f"Connector [{i}]: conduit Ø{diam}mm not standard",
                        field_path=f"connectors[{i}].dimensions.diameter",
                        suggestion=f"Nearest standard: Ø{nearest}mm",
                    ))

            # Rectangular duct checks
            if domain == "DomainHvac" and shape == "Rectangular":
                w = dims.get("width")
                h = dims.get("height")
                if w and h:
                    if h > w:
                        r.add(ValidationResult(
                            rule="RECT_DUCT_ORIENTATION",
                            severity=Severity.WARNING,
                            message=f"Connector [{i}]: height ({h}) > width ({w}) — usually W ≥ H",
                            field_path=f"connectors[{i}].dimensions",
                            suggestion="Swap width and height",
                        ))
                    aspect = w / h if h > 0 else 999
                    if aspect > 4:
                        r.add(ValidationResult(
                            rule="RECT_DUCT_ASPECT_RATIO",
                            severity=Severity.WARNING,
                            message=f"Connector [{i}]: aspect ratio {aspect:.1f}:1 exceeds 4:1 max",
                            field_path=f"connectors[{i}].dimensions",
                            suggestion="Keep aspect ratio ≤ 4:1 per SMACNA guidelines",
                        ))

    def _check_connector_positions(self, f: dict, r: ValidationReport):
        connectors = f.get("connectors", [])
        positions = []
        for i, conn in enumerate(connectors):
            pos = tuple(conn.get("position", [0, 0, 0]))
            positions.append(pos)

        # Check for overlapping connectors
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(positions[i], positions[j])))
                if dist < 1.0:  # < 1mm apart
                    r.add(ValidationResult(
                        rule="OVERLAPPING_CONNECTORS",
                        severity=Severity.WARNING,
                        message=f"Connectors [{i}] and [{j}] are less than 1mm apart",
                        field_path=f"connectors[{i}].position",
                        suggestion="Ensure connectors have distinct positions",
                    ))

    def _check_flow_balance(self, f: dict, r: ValidationReport):
        """Check that HVAC/piping elements have balanced in/out flows."""
        connectors = f.get("connectors", [])
        category = f.get("category", "")

        # Only check for equipment that should have balanced flow
        balanced_categories = {"Mechanical Equipment", "Duct Accessories"}
        if category not in balanced_categories:
            return

        duct_in = sum(1 for c in connectors
                     if c.get("domain") == "DomainHvac" and c.get("flow_direction") == "In")
        duct_out = sum(1 for c in connectors
                      if c.get("domain") == "DomainHvac" and c.get("flow_direction") == "Out")

        if duct_in > 0 and duct_out == 0 and category == "Mechanical Equipment":
            r.add(ValidationResult(
                rule="DUCT_FLOW_BALANCE",
                severity=Severity.INFO,
                message="Equipment has duct inlet(s) but no outlet — is this intentional?",
                field_path="connectors",
                suggestion="Most HVAC equipment needs both supply and return connections",
            ))

        # For piping: check paired supply/return
        pipe_systems = [c.get("system_type") for c in connectors if c.get("domain") == "DomainPiping"]
        paired = {
            "ChilledWater": "ChilledWater",
            "HydronicSupply": "HydronicReturn",
            "HydronicReturn": "HydronicSupply",
        }
        for sys in pipe_systems:
            if sys in paired and paired[sys] not in pipe_systems:
                r.add(ValidationResult(
                    rule="PIPE_PAIR_MISSING",
                    severity=Severity.WARNING,
                    message=f"System '{sys}' usually requires paired '{paired[sys]}' connection",
                    field_path="connectors",
                ))

    # ── Parameter validation ─────────────────

    def _check_parameters(self, f: dict, r: ValidationReport):
        params = f.get("calculation_params", [])
        param_names = set()

        for i, p in enumerate(params):
            name = p.get("name", "")
            if not name:
                r.add(ValidationResult(
                    rule="PARAM_NAME_MISSING",
                    severity=Severity.ERROR,
                    message=f"Parameter [{i}] has no name",
                    field_path=f"calculation_params[{i}].name",
                ))
                continue

            if name in param_names:
                r.add(ValidationResult(
                    rule="DUPLICATE_PARAM",
                    severity=Severity.ERROR,
                    message=f"Duplicate parameter name: '{name}'",
                    field_path=f"calculation_params[{i}].name",
                ))
            param_names.add(name)

            # Check physical limits
            ptype = p.get("param_type", "")
            default = p.get("default_value")
            min_val = p.get("min_value")
            max_val = p.get("max_value")

            if ptype in PARAM_PHYSICAL_LIMITS and default is not None:
                limits = PARAM_PHYSICAL_LIMITS[ptype]
                if default < limits["min"] or default > limits["max"]:
                    r.add(ValidationResult(
                        rule="PARAM_OUT_OF_RANGE",
                        severity=Severity.ERROR,
                        message=(f"Parameter '{name}': default {default} outside physical limits "
                                f"[{limits['min']}, {limits['max']}] {limits['unit']}"),
                        field_path=f"calculation_params[{i}].default_value",
                    ))

            if min_val is not None and max_val is not None and min_val > max_val:
                r.add(ValidationResult(
                    rule="PARAM_RANGE_INVERTED",
                    severity=Severity.ERROR,
                    message=f"Parameter '{name}': min ({min_val}) > max ({max_val})",
                    field_path=f"calculation_params[{i}]",
                ))

    # ── Formula validation ───────────────────

    def _check_formulas(self, f: dict, r: ValidationReport):
        formulas = f.get("formulas", [])
        param_names = {p.get("name") for p in f.get("calculation_params", [])}

        for i, formula in enumerate(formulas):
            target = formula.get("target_param", "")
            deps = formula.get("dependencies", [])
            expr = formula.get("expression", "")

            if not target:
                r.add(ValidationResult(
                    rule="FORMULA_NO_TARGET",
                    severity=Severity.ERROR,
                    message=f"Formula [{i}] has no target parameter",
                    field_path=f"formulas[{i}].target_param",
                ))

            if target and target not in param_names:
                r.add(ValidationResult(
                    rule="FORMULA_TARGET_NOT_FOUND",
                    severity=Severity.WARNING,
                    message=f"Formula target '{target}' not found in parameters",
                    field_path=f"formulas[{i}].target_param",
                    suggestion=f"Available params: {sorted(param_names)[:10]}",
                ))

            for dep in deps:
                if dep not in param_names:
                    r.add(ValidationResult(
                        rule="FORMULA_DEP_NOT_FOUND",
                        severity=Severity.WARNING,
                        message=f"Formula [{i}] dependency '{dep}' not found in parameters",
                        field_path=f"formulas[{i}].dependencies",
                    ))

            if not expr:
                r.add(ValidationResult(
                    rule="FORMULA_EMPTY",
                    severity=Severity.ERROR,
                    message=f"Formula [{i}] has empty expression",
                    field_path=f"formulas[{i}].expression",
                ))

            # Check for circular dependencies
            if target in deps:
                r.add(ValidationResult(
                    rule="FORMULA_CIRCULAR",
                    severity=Severity.ERROR,
                    message=f"Formula [{i}]: target '{target}' is in its own dependencies",
                    field_path=f"formulas[{i}]",
                ))

    # ── Classification validation ────────────

    def _check_classification(self, f: dict, r: ValidationReport):
        classification = f.get("classification", {})
        category = f.get("category", "")

        ifc_class = classification.get("ifc_class")
        if ifc_class and ifc_class in IFC_CATEGORY_MAP:
            expected_cats = IFC_CATEGORY_MAP[ifc_class]
            if category and category not in expected_cats:
                r.add(ValidationResult(
                    rule="IFC_CATEGORY_MISMATCH",
                    severity=Severity.WARNING,
                    message=f"IFC class '{ifc_class}' typically maps to {expected_cats}, not '{category}'",
                    field_path="classification.ifc_class",
                ))

        if not classification.get("ifc_class"):
            r.add(ValidationResult(
                rule="MISSING_IFC_CLASS",
                severity=Severity.INFO,
                message="No IFC class specified — recommended for IFC export",
                field_path="classification.ifc_class",
            ))

        if not classification.get("omniclass_table"):
            r.add(ValidationResult(
                rule="MISSING_OMNICLASS",
                severity=Severity.INFO,
                message="No OmniClass code — recommended for classification",
                field_path="classification.omniclass_table",
            ))

    # ── Host type validation ─────────────────

    def _check_host_type(self, f: dict, r: ValidationReport):
        host = f.get("host_type", "")
        template = f.get("template_file", "")

        if host and template:
            expected_keywords = HOST_TEMPLATE_MAP.get(host, [])
            if expected_keywords and expected_keywords != [""]:
                if not any(kw.lower() in template.lower() for kw in expected_keywords):
                    r.add(ValidationResult(
                        rule="HOST_TEMPLATE_MISMATCH",
                        severity=Severity.WARNING,
                        message=f"Host type '{host}' but template '{template}' doesn't match",
                        field_path="template_file",
                        suggestion=f"Expected template with: {expected_keywords}",
                    ))

    # ── Type variant validation ──────────────

    def _check_type_variants(self, f: dict, r: ValidationReport):
        types = f.get("family_types", [])
        if not types:
            r.add(ValidationResult(
                rule="NO_TYPE_VARIANTS",
                severity=Severity.INFO,
                message="No type variants defined — family will have single default type",
                field_path="family_types",
            ))
            return

        names = [t.get("name", "") for t in types]
        if len(names) != len(set(names)):
            r.add(ValidationResult(
                rule="DUPLICATE_TYPE_NAME",
                severity=Severity.ERROR,
                message="Type variant names must be unique",
                field_path="family_types",
            ))

        for i, t in enumerate(types):
            if not t.get("name"):
                r.add(ValidationResult(
                    rule="TYPE_NO_NAME",
                    severity=Severity.ERROR,
                    message=f"Type variant [{i}] has no name",
                    field_path=f"family_types[{i}].name",
                ))

    # ── Category-Domain consistency ──────────

    def _check_category_domain(self, f: dict, r: ValidationReport):
        category = f.get("category", "")
        connectors = f.get("connectors", [])

        if category not in CATEGORY_DOMAIN_MAP:
            return  # Unknown category — skip

        allowed_domains = CATEGORY_DOMAIN_MAP[category]
        for i, conn in enumerate(connectors):
            domain = conn.get("domain", "")
            if domain and domain not in allowed_domains:
                r.add(ValidationResult(
                    rule="CATEGORY_DOMAIN_MISMATCH",
                    severity=Severity.ERROR,
                    message=(f"Category '{category}' does not allow connector domain "
                            f"'{domain}' (connector [{i}])"),
                    field_path=f"connectors[{i}].domain",
                    suggestion=f"Allowed domains for {category}: {sorted(allowed_domains)}",
                ))


# ─────────────────────────────────────────────
# Batch Validation
# ─────────────────────────────────────────────

def validate_dataset(families: list[dict]) -> dict:
    """Validate an entire dataset and return statistics."""
    validator = MEPValidator()
    reports = [validator.validate(f) for f in families]

    total = len(reports)
    valid = sum(1 for r in reports if r.is_valid)
    all_errors = [e for r in reports for e in r.errors]
    all_warnings = [w for r in reports for w in r.warnings]

    # Error frequency
    error_freq = {}
    for e in all_errors:
        error_freq[e.rule] = error_freq.get(e.rule, 0) + 1

    warning_freq = {}
    for w in all_warnings:
        warning_freq[w.rule] = warning_freq.get(w.rule, 0) + 1

    return {
        "total_families": total,
        "valid_families": valid,
        "invalid_families": total - valid,
        "validity_rate": f"{valid/total*100:.1f}%" if total else "N/A",
        "total_errors": len(all_errors),
        "total_warnings": len(all_warnings),
        "error_frequency": dict(sorted(error_freq.items(), key=lambda x: -x[1])),
        "warning_frequency": dict(sorted(warning_freq.items(), key=lambda x: -x[1])),
        "reports": reports,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from mep_generator import MEPFamilyGenerator

    gen = MEPFamilyGenerator(seed=42)
    dataset = gen.generate_dataset(n_per_template=2)

    stats = validate_dataset(dataset)

    print("=" * 60)
    print("MEP VALIDATION REPORT")
    print("=" * 60)
    print(f"Total families: {stats['total_families']}")
    print(f"Valid: {stats['valid_families']} ({stats['validity_rate']})")
    print(f"Invalid: {stats['invalid_families']}")
    print(f"Total errors: {stats['total_errors']}")
    print(f"Total warnings: {stats['total_warnings']}")

    if stats["error_frequency"]:
        print(f"\nTop errors:")
        for rule, count in list(stats["error_frequency"].items())[:10]:
            print(f"  {count:3d}x {rule}")

    if stats["warning_frequency"]:
        print(f"\nTop warnings:")
        for rule, count in list(stats["warning_frequency"].items())[:10]:
            print(f"  {count:3d}x {rule}")

    # Show details for invalid families
    print(f"\n{'='*60}")
    for report in stats["reports"]:
        if not report.is_valid or report.warnings:
            print(report.summary())
            print()
