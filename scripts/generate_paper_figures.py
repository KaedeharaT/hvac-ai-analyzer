"""Generate publication figures and tables from a finalized experiment artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Preserve CJK point/equipment labels in publication exports on the supported
# Windows research environment while retaining a portable Latin fallback.
plt.rcParams["font.sans-serif"] = ["Yu Gothic", "Meiryo", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.experiment; output = args.output or root / "figures"; output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "energy_summary.json").read_text(encoding="utf-8"))
    for name, chart in summary.get("charts", {}).items():
        series = chart.get("series", [])
        if not series:
            continue
        figure, axis = plt.subplots(figsize=(7, 3.5))
        for item in series:
            data = pd.DataFrame(item.get("data", []))
            if {"time", "value"} <= set(data.columns):
                axis.plot(pd.to_datetime(data["time"], errors="coerce"), data["value"], label=item.get("name", "series"))
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Time")
        axis.set_ylabel(chart.get("unit", "value"))
        if len(series) > 1:
            axis.legend()
        figure.tight_layout(); figure.savefig(output / f"figure_{name}.pdf"); figure.savefig(output / f"figure_{name}.png", dpi=300); plt.close(figure)
    for name in ("semantic_per_class.csv", "kpi_summary.csv"):
        path = root / name
        if path.exists():
            try:
                table = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                # A capability-driven experiment may legitimately have no KPI
                # table.  Publication export should record that absence rather
                # than failing after valid figures have already been produced.
                continue
            table.to_csv(output / f"table_{path.stem}.csv", index=False)
            (output / f"table_{path.stem}.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
