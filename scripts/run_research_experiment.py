"""Run one immutable, UI-independent BuildingAI research experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_ai.research.experiment_runner import ResearchExperimentRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Immutable JSON configuration or an experiment config snapshot")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts" / "experiments")
    parser.add_argument("--experiment-id")
    parser.add_argument("--allow-dirty", action="store_true", help="Create a DRAFT/explicitly non-clean research artifact only")
    args = parser.parse_args()
    config = ResearchExperimentRunner.load_config(args.config)
    runner = ResearchExperimentRunner(args.artifact_root, repository_root=ROOT)
    result = runner.run(config, experiment_id=args.experiment_id, allow_dirty=args.allow_dirty)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
