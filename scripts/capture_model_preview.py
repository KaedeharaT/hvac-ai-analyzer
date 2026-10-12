"""Capture the real optional viewer using only self-generated synthetic inputs.

No historical/user model is accepted by this publication helper. The OSM and
BEMS snapshots remain in ignored artifacts; only the native GUI PNG is copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def synthetic_building() -> bytes:
    def handle(name):
        return "{" + str(uuid.uuid5(uuid.NAMESPACE_URL, "buildingai/synthetic-preview/" + name)) + "}"

    records = [f"OS:Version,{handle('version')},3.7.0;"]

    def polygon(name, category, space, vertices, parent=None):
        if parent:
            fields = ["OS:SubSurface", handle(name), name, category, "", handle(parent), "", "", "", "1", "4"]
        else:
            fields = ["OS:Surface", handle(name), name, category, "", handle(space),
                      "Outdoors", "", "SunExposed", "WindExposed", "", "4"]
        records.append(",".join(fields + [str(value) for vertex in vertices for value in vertex]) + ";")

    for floor in range(3):
        space = f"Synthetic level {floor + 1}"
        bottom, top = floor * 3.2, (floor + 1) * 3.2
        records.append(f"OS:Space,{handle(space)},{space},,,,0,0,0,0,;")
        surfaces = {
            "Floor": ("Floor", [(0,0,bottom),(0,8,bottom),(12,8,bottom),(12,0,bottom)]),
            "Ceiling": ("RoofCeiling", [(0,0,top),(12,0,top),(12,8,top),(0,8,top)]),
            "South wall": ("Wall", [(0,0,bottom),(12,0,bottom),(12,0,top),(0,0,top)]),
            "East wall": ("Wall", [(12,0,bottom),(12,8,bottom),(12,8,top),(12,0,top)]),
            "North wall": ("Wall", [(12,8,bottom),(0,8,bottom),(0,8,top),(12,8,top)]),
            "West wall": ("Wall", [(0,8,bottom),(0,0,bottom),(0,0,top),(0,8,top)]),
        }
        for label, (category, vertices) in surfaces.items():
            polygon(f"{space} - {label}", category, space, vertices)
        for number, x in enumerate((1, 4.5, 8)):
            polygon(f"{space} - South window {number+1}", "FixedWindow", space,
                    [(x,0,bottom+.9),(x+2.5,0,bottom+.9),(x+2.5,0,bottom+2.4),(x,0,bottom+2.4)],
                    parent=f"{space} - South wall")
        for number, y in enumerate((1, 4.5)):
            polygon(f"{space} - East window {number+1}", "FixedWindow", space,
                    [(12,y,bottom+.9),(12,y+2.5,bottom+.9),(12,y+2.5,bottom+2.4),(12,y,bottom+2.4)],
                    parent=f"{space} - East wall")
    return ("! Synthetic geometry for software demonstration only.\n" + "\n".join(records)).encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="artifacts/model-preview-public")
    parser.add_argument("--output", default="docs/images/model-viewer.png")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve(); workspace.mkdir(parents=True, exist_ok=True)
    data = synthetic_building()
    model = workspace / "synthetic-building.osm"; model.write_bytes(data)
    subprocess.run([sys.executable, str(ROOT / "scripts/qa_model_viewer.py"),
                    "--model", str(model), "--workspace", str(workspace / "gui"), "--scale", "1"],
                   cwd=ROOT, check=True, timeout=240)
    report = json.loads((workspace / "gui/report.json").read_text(encoding="utf-8"))
    assert report["model_hash"] == hashlib.sha256(data).hexdigest()
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(workspace / "gui/en_US-equipment-details.png", destination)
    print("Published native screenshot with synthetic geometry and synthetic BEMS:", destination)


if __name__ == "__main__":
    main()
