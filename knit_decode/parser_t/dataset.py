from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TypedDict

from PIL import Image


JsonObject = dict[str, object]


class ParserSample(TypedDict):
    sample_id: str
    image_path: str
    topo_path: str
    category: str


class ParserBatchItem(TypedDict):
    sample_id: str
    category: str
    image: object
    topo_ids: object
    rows: int
    columns: int


@dataclass(frozen=True)
class TopologyTarget:
    sample_id: str
    category: str
    image_path: Path
    topo_path: Path
    rows: int
    columns: int
    topo_ids: list[list[int]]


def _read_json(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_topology_target(sample: ParserSample, root: Path) -> TopologyTarget:
    image_path = root / sample["image_path"]
    topo_path = root / sample["topo_path"]
    payload = _read_json(topo_path)
    rows = payload.get("rows")
    columns = payload.get("columns")
    grid = payload.get("grid")
    if not isinstance(rows, int) or not isinstance(columns, int) or not isinstance(grid, list):
        raise ValueError(f"Invalid topology grid payload at {topo_path}")

    topo_ids: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list):
            raise ValueError(f"Invalid topology row in {topo_path}")
        parsed_row: list[int] = []
        for token in row:
            if not isinstance(token, int):
                raise ValueError(f"Invalid topology token in {topo_path}: {token!r}")
            parsed_row.append(token)
        topo_ids.append(parsed_row)

    if len(topo_ids) != rows or any(len(row) != columns for row in topo_ids):
        raise ValueError(f"Topology shape mismatch in {topo_path}")

    return TopologyTarget(
        sample_id=sample["sample_id"],
        category=sample["category"],
        image_path=image_path,
        topo_path=topo_path,
        rows=rows,
        columns=columns,
        topo_ids=topo_ids,
    )


def load_parser_manifest(path: str | Path) -> list[ParserSample]:
    manifest_path = Path(path)
    rows: list[ParserSample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object line in {manifest_path}")
        required = ("sample_id", "image_path", "topo_path", "category")
        if any(not isinstance(payload.get(key), str) for key in required):
            raise ValueError(f"Invalid parser sample entry in {manifest_path}: {payload!r}")
        rows.append(
            {
                "sample_id": str(payload["sample_id"]),
                "image_path": str(payload["image_path"]),
                "topo_path": str(payload["topo_path"]),
                "category": str(payload["category"]),
            }
        )
    return rows


def build_parser_manifest_from_dataset_complete(dataset_root: str | Path, output_path: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    simulation_root = dataset_root / "simulation images"
    stitch_root = dataset_root / "stitch code patterns"
    manifest_rows: list[dict[str, object]] = []

    for category_dir in sorted(simulation_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for image_path in sorted(category_dir.glob("*.png")):
            topo_name = f"{image_path.stem}_topo.json"
            topo_path = output_path.parent / "targets" / category / topo_name
            stitch_candidate = stitch_root / category / f"{image_path.stem}_resized.png"
            if not stitch_candidate.exists():
                continue
            manifest_rows.append(
                {
                    "sample_id": f"{category}/{image_path.stem}",
                    "category": category,
                    "image_path": str(image_path.relative_to(output_path.parent)).replace("\\", "/"),
                    "topo_path": str(topo_path.relative_to(output_path.parent)).replace("\\", "/"),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows)
    if text:
        text += "\n"
    output_path.write_text(text, encoding="utf-8")
    return output_path


class SimulationTopologyDataset:
    def __init__(self, manifest_path: str | Path, root: str | Path | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.samples = load_parser_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TopologyTarget:
        sample = self.samples[index]
        return _load_topology_target(sample, self.root)

    @staticmethod
    def load_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

