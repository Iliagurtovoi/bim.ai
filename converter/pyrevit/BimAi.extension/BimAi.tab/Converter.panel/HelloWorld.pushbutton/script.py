# -*- coding: utf-8 -*-
"""
BIM.AI — Hello World
====================
Smoke test: открывает шаблон Metric Generic Model ceiling based.rft,
сохраняет как hello_output.rfa.

Если эта кнопка срабатывает и создаёт файл — значит окружение
готово для разработки MVP конвертера.
"""

import os
from pyrevit import revit, DB, script, forms

__title__ = "Hello World"
__doc__   = "Smoke test: open .rft template and save as .rfa"


TEMPLATES_DIR = os.environ.get(
    "REVIT_TEMPLATES_DIR",
    r"C:\ProgramData\Autodesk\RVT 2025\Family Templates\English"
)
TEMPLATE_NAME = "Metric Generic Model ceiling based.rft"

OUTPUT_DIR  = os.path.expanduser(r"~\Documents\bim_ai_output")
OUTPUT_NAME = "hello_output.rfa"


def main():
    logger = script.get_logger()
    app = __revit__.Application

    template_path = os.path.join(TEMPLATES_DIR, TEMPLATE_NAME)
    if not os.path.isfile(template_path):
        forms.alert(
            "Template not found:\n{}\n\n"
            "Check REVIT_TEMPLATES_DIR or install Family Templates."
            .format(template_path),
            exitscript=True
        )

    logger.info("Opening template: {}".format(template_path))
    fam_doc = app.NewFamilyDocument(template_path)
    if fam_doc is None:
        forms.alert("NewFamilyDocument returned None", exitscript=True)

    logger.info("Category: {}".format(fam_doc.OwnerFamily.FamilyCategory.Name))

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    if os.path.isfile(output_path):
        os.remove(output_path)

    save_opts = DB.SaveAsOptions()
    save_opts.OverwriteExistingFile = True
    fam_doc.SaveAs(output_path, save_opts)
    fam_doc.Close(False)

    size_kb = os.path.getsize(output_path) // 1024
    forms.alert(
        "[OK] Saved:\n{}\n\nFile size: {} KB".format(output_path, size_kb)
    )
    logger.info("Saved: {} ({} KB)".format(output_path, size_kb))


if __name__ == "__main__":
    main()