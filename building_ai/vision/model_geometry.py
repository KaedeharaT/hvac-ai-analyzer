"""Read a deliberately limited, documented subset of existing OSM geometry.

No OpenStudio/Revit execution, external references, inferred dimensions or
pixel-to-model conversion. Nonidentity space transforms require a separate,
validated adapter and are rejected here.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

import numpy as np

MAX_BYTES = 20 * 1024 * 1024
ADAPTER_VERSION = "osm-surfaces-1"
SUPPORTED_VERSIONS = {"3.6.1", "3.7.0"}


def _polygon(values: list[str]) -> list[list[float]]:
    if len(values) % 3 or not 9 <= len(values) <= 384:
        raise ValueError("Unsupported polygon vertex count")
    points = np.array([float(v) for v in values], dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all() or np.abs(points).max() > 1e7:
        raise ValueError("Invalid model coordinate")
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3 or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Repeated or missing polygon vertices")
    normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
    length = np.linalg.norm(normal)
    if length < 1e-10:
        raise ValueError("Degenerate polygon")
    normal /= length
    tolerance = max(1., float(np.ptp(points, axis=0).max())) * 1e-7
    if np.abs((points - points[0]) @ normal).max() > tolerance:
        raise ValueError("Nonplanar polygon")
    flat = np.delete(points, int(np.argmax(np.abs(normal))), axis=1)
    def turn(a, b, c):
        ab, ac = b-a, c-a
        return float(ab[0]*ac[1]-ab[1]*ac[0])
    for i in range(len(flat)):
        for j in range(i+2, len(flat)):
            if i == 0 and j == len(flat)-1:
                continue
            a, b, c, d = flat[i], flat[(i+1)%len(flat)], flat[j], flat[(j+1)%len(flat)]
            if turn(a,b,c)*turn(a,b,d) < 0 and turn(c,d,a)*turn(c,d,b) < 0:
                raise ValueError("Self-intersecting polygon")
    return points.tolist()


def read_osm(data: bytes) -> dict:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Model file size outside preview limits")
    text = data.decode("utf-8-sig")
    # OSM/IDD comments end at the newline. No fields are executed or dereferenced.
    clean = re.sub(r"!.*", "", text)
    objects = defaultdict(list)
    for record in clean.split(";"):
        if not record.strip():
            continue
        fields = [field.strip() for field in record.split(",")]
        objects[fields[0]].append(fields[1:])
    versions = objects["OS:Version"]
    if len(versions) != 1 or len(versions[0]) < 2 or versions[0][1] not in SUPPORTED_VERSIONS:
        raise ValueError("Preview supports OSM 3.6.1 and 3.7.0 only")
    spaces = {}
    for fields in objects["OS:Space"]:
        if len(fields) < 2 or fields[0] in spaces:
            raise ValueError("Invalid or duplicate space")
        # Defaults are zero in the versioned OpenStudio IDD, not inferred from images.
        transform = (fields + [""]*9)[5:9]
        if any(not np.isfinite(float(v or 0)) or float(v or 0) != 0 for v in transform):
            raise ValueError("Nonidentity space origin/rotation requires a validated transform")
        spaces[fields[0]] = fields[1]
    surfaces = {f[0]: f for f in objects["OS:Surface"] if len(f) >= 11}
    elements = []
    seen = set()
    for kind, offset, count_index in (("OS:Surface", 11, 10), ("OS:SubSurface", 10, 9)):
        for fields in objects[kind]:
            if len(fields) < offset+9 or not fields[0] or fields[0] in seen:
                raise ValueError("Invalid or duplicate surface")
            seen.add(fields[0])
            space_id = fields[4] if kind == "OS:Surface" else (surfaces.get(fields[4]) or [""]*5)[4]
            if space_id not in spaces:
                raise ValueError("Surface has no resolvable space")
            points = _polygon(fields[offset:])
            if fields[count_index].lower() not in ("", "autocalculate") and float(fields[count_index]) != len(points):
                raise ValueError("Declared vertex count does not match geometry")
            if kind == "OS:SubSurface" and float(fields[8] or 1) != 1:
                raise ValueError("Repeated openings do not define separate geometric positions")
            elements.append({"object_id": fields[0], "name": fields[1],
                "category": fields[2] or "Unknown",
                "source_type": kind, "space_id": space_id, "space_name": spaces[space_id],
                "parent_id": fields[4] if kind == "OS:SubSurface" else None,
                "vertices": points})
    if not elements or len(elements) > 20000:
        raise ValueError("No supported surfaces, or too many surfaces")
    points = np.array([p for e in elements for p in e["vertices"]])
    return {"schema_version": 1, "source_format": "OpenStudio OSM", "source_version": versions[0][1],
        "model_hash": hashlib.sha256(data).hexdigest(), "adapter_version": ADAPTER_VERSION,
        "coordinate_system": "model_local", "unit": "m", "up_axis": "Z",
        "transform_method": "identity_zero_space_origins_and_rotations",
        "unit_reference": "OpenStudio versioned IDD OS:Surface/OS:SubSurface vertex units",
        "engineering_validation": "not_verified_as_built", "elements": elements,
        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "categories": dict(Counter(e["category"] for e in elements)),
        "omitted_geometry": {k: len(v) for k, v in objects.items()
            if any(w in k for w in ("ShadingSurface", "InteriorPartitionSurface")) and v},
        "boundary": "Surface polygons only; no inferred solids, opening cuts, equipment geometry or topology"}
