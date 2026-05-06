from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import TypedDict

from PIL import Image


JsonObject = dict[str, object]


class ParserSample(TypedDict):
    sample_id: str
    image_path: str
    target_path: str
    category: str


@dataclass(frozen=True)
class SegmentationTarget:
    sample_id: str
    category: str
    image_path: Path
    target_path: Path


_HEM_PREFIX_RE = re.compile(r"^(\d+[A-Za-z]*)")


def _normalize_sim_stem(category: str, stem: str) -> str:
    if category == "Hem":
        match = _HEM_PREFIX_RE.match(stem)
        if match:
            return match.group(1)
    return stem


def load_parser_manifest(path: str | Path) -> list[ParserSample]:
    manifest_path = Path(path)
    rows: list[ParserSample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object line in {manifest_path}")
        required = ("sample_id", "image_path", "target_path", "category")
        if any(not isinstance(payload.get(key), str) for key in required):
            raise ValueError(f"Invalid parser sample entry in {manifest_path}: {payload!r}")
        rows.append(
            {
                "sample_id": str(payload["sample_id"]),
                "image_path": str(payload["image_path"]),
                "target_path": str(payload["target_path"]),
                "category": str(payload["category"]),
            }
        )
    return rows


def build_parser_manifest_from_dataset_complete(dataset_root: str | Path, output_path: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    manifest_root = output_path.parent.resolve()
    simulation_root = dataset_root / "simulation images"
    stitch_root = dataset_root / "stitch code patterns"
    manifest_rows: list[dict[str, object]] = []

    for category_dir in sorted(simulation_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        stitch_dir = stitch_root / category
        if not stitch_dir.exists():
            continue
        stitch_index = {path.stem: path for path in stitch_dir.glob("*.png")}
        for image_path in sorted(category_dir.glob("*.png")):
            normalized = _normalize_sim_stem(category, image_path.stem)
            stitch_candidate = stitch_index.get(f"{normalized}_resized")
            if stitch_candidate is None:
                continue
            manifest_rows.append(
                {
                    "sample_id": f"{category}/{image_path.stem}",
                    "category": category,
                    "image_path": os.path.relpath(image_path.resolve(), manifest_root).replace("\\", "/"),
                    "target_path": os.path.relpath(stitch_candidate.resolve(), manifest_root).replace("\\", "/"),
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

    def __getitem__(self, index: int) -> SegmentationTarget:
        sample = self.samples[index]
        return SegmentationTarget(
            sample_id=sample["sample_id"],
            category=sample["category"],
            image_path=self.root / sample["image_path"],
            target_path=self.root / sample["target_path"],
        )

    @staticmethod
    def load_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")
