# BIM.AI Converter (Звено 4)

JSON → .rfa conversion via Revit API.

## Status

🔄 Setup phase: pyRevit extension scaffold, template registry inventory.

## Structure

- `pyrevit/BimAi.extension/` — pyRevit extension, loaded into Revit via Custom Extension Directories
- `templates/template_registry.json` — mapping from MEP category+host to .rft template file
- `.env.example` — template for local config (copy to `.env`, adjust paths)

## Setup

1. Install Revit 2025 with Family Templates content.
2. Install pyRevit: https://github.com/pyrevitlabs/pyRevit/releases
3. In Revit: pyRevit tab → Settings → Custom Extension Directories → Add `<REPO>/converter/pyrevit/`
4. Reload pyRevit. The `BimAi` tab should appear on the ribbon.
5. Copy `.env.example` to `.env`, adjust paths.
6. Click **BimAi → Converter → Hello World** to verify the environment.

## MVP scope (next)

Generate a ceiling-mounted round diffuser `.rfa` from JSON spec:
- geometry: cylinder body + circular flange
- one HVAC connector (SupplyAir, Round, Ø200mm)
- three family parameters with one formula
- one family type (Ø200 / 150 L/s)
- smoke test via `../rfa_parser.py` (if available) or manual load in Revit sandbox