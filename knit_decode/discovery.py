from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_CATEGORY_SUFFIX_RE = re.compile(r"\d+$")


@dataclass(frozen=True)
class StitchSample:
    category: str
    source_path: Path
    source_stem: str
    pairing_key: str
    simulation_path: Path | None


def strip_resized_suffix(stem: str) -> str:
    return stem[:-8] if stem.lower().endswith("_resized") else stem


def normalize_pairing_key(stem: str) -> str:
    base = strip_resized_suffix(Path(stem).stem)
    token = base.split("_", 1)[0]
    match = re.match(r"^0*(\d+)([A-Za-z]*)$", token)
    if match:
        number, suffix = match.groups()
        return f"{int(number)}{suffix.upper()}"
    cleaned = re.sub(r"[^0-9A-Za-z]+", "", token)
    return cleaned.upper()


def _simulation_category_candidates(category: str) -> list[str]:
    candidates = [category]
    simplified = _CATEGORY_SUFFIX_RE.sub("", category)
    if simplified and simplified not in candidates:
        candidates.append(simplified)
    return candidates


def _build_simulation_index(simulation_root: Path, category: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for candidate_category in _simulation_category_candidates(category):
        category_dir = simulation_root / candidate_category
        if not category_dir.exists():
            continue
        for image_path in sorted(category_dir.glob("*.png")):
            pairing_key = normalize_pairing_key(image_path.stem)
            if pairing_key not in index:
                index[pairing_key] = image_path
    return index


def discover_samples(dataset_root: str | Path, categories: list[str] | None = None) -> list[StitchSample]:
    dataset_root = Path(dataset_root)
    stitch_root = dataset_root / "stitch code patterns"
    simulation_root = dataset_root / "simulation images"
    if not stitch_root.exists():
        return []

    category_names = categories or sorted(path.name for path in stitch_root.iterdir() if path.is_dir())
    samples: list[StitchSample] = []
    for category in category_names:
        category_dir = stitch_root / category
        if not category_dir.exists():
            continue
        simulation_index = _build_simulation_index(simulation_root, category)
        for image_path in sorted(category_dir.glob("*.png")):
            source_stem = strip_resized_suffix(image_path.stem)
            pairing_key = normalize_pairing_key(source_stem)
            samples.append(
                StitchSample(
                    category=category,
                    source_path=image_path,
                    source_stem=source_stem,
                    pairing_key=pairing_key,
                    simulation_path=simulation_index.get(pairing_key),
                )
            )
    return samples
