from dataclasses import dataclass
from pathlib import PurePath
from uuid import NAMESPACE_URL, uuid5


def make_point_id(
    project_id: str, source_file: str | None, sheet: str | None, raw_name: str
) -> str:
    """Return a stable identity for one source column within a project."""
    source = str(PurePath(source_file or "<unknown-source>")).replace("\\", "/").casefold()
    key = "\x1f".join((project_id, source, sheet or "<no-sheet>", raw_name))
    return str(uuid5(NAMESPACE_URL, f"building-ai-point:{key}"))


@dataclass(slots=True)
class Point:
    project_id: str
    raw_name: str
    source_file: str | None = None
    sheet: str | None = None
    point_id: str = ""
    column_index: int | None = None
    equipment_id: str | None = None
    semantic_result_id: str | None = None

    def __post_init__(self) -> None:
        if not self.point_id:
            self.point_id = make_point_id(
                self.project_id, self.source_file, self.sheet, self.raw_name
            )
