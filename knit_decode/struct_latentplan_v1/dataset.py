from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast


class LatentPlanSample(TypedDict):
    sample_id: str
    category: str
    input_path: str
    target_path: str
    index_path: str


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        data = importlib.import_module("torch.utils.data")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_latentplan_v1 dataset. Install with `pip install -e .[train]`.") from error
    return torch, data


def load_manifest(path: str | Path) -> list[LatentPlanSample]:
    manifest_path = Path(path)
    rows: list[LatentPlanSample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object line in {manifest_path}")
        row = {}
        for key in ["sample_id", "category", "input_path", "target_path", "index_path"]:
            value = payload.get(key)
            if not isinstance(value, str):
                raise ValueError(f"Missing {key!r} in manifest row: {payload!r}")
            row[key] = value
        rows.append(cast(LatentPlanSample, row))
    return rows


class LatentPlanDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        palette_path: str | Path | None,
        plan_cache_path: str | Path,
        category_to_id: dict[str, int] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.samples = load_manifest(self.manifest_path)
        cache_payload = _require_torch()[0].load(Path(plan_cache_path), map_location="cpu")
        self.cache_payload = cache_payload
        self.cache_meta = cache_payload["meta"]
        self.category_fg_stats = cache_payload.get("category_fg_stats")
        self.category_occ_stats = cache_payload.get("category_occ_stats")
        if not isinstance(self.category_fg_stats, dict) or not self.category_fg_stats:
            raise ValueError(f"category_fg_stats missing or empty in plan cache: {plan_cache_path}")
        if not isinstance(self.category_occ_stats, dict) or not self.category_occ_stats:
            raise ValueError(f"category_occ_stats missing or empty in plan cache: {plan_cache_path}")
        self.cache_by_id = {entry["sample_id"]: entry for entry in cache_payload["items"]}
        categories = sorted({sample["category"] for sample in self.samples})
        self.category_to_id = category_to_id or {category: index for index, category in enumerate(categories)}
        self.num_classes = int(self.cache_meta.get("num_classes", 17))
        self.max_num_modes_per_category = int(self.cache_meta.get("max_num_modes_per_category", self.cache_meta.get("num_modes_per_category", 16)))
        self.palette_path = str(palette_path) if palette_path is not None else None
        self._validate_alignment()

    def _validate_alignment(self) -> None:
        manifest_ids = {sample["sample_id"] for sample in self.samples}
        cache_ids = set(self.cache_by_id)
        missing_cache_ids = sorted(manifest_ids - cache_ids)
        extra_cache_ids = sorted(cache_ids - manifest_ids)
        if missing_cache_ids:
            preview = missing_cache_ids[:8]
            raise KeyError(
                f"Plan cache is missing sample ids for manifest={self.manifest_path}: {preview} "
                f"(missing={len(missing_cache_ids)})"
            )
        if extra_cache_ids:
            preview = extra_cache_ids[:8]
            raise ValueError(
                f"Plan cache contains sample ids not present in manifest={self.manifest_path}: {preview} "
                f"(extra={len(extra_cache_ids)})"
            )
        if len(self.samples) != len(self.cache_by_id):
            raise ValueError(
                f"Manifest/cache length mismatch for {self.manifest_path}: "
                f"manifest_rows={len(self.samples)} cache_rows={len(self.cache_by_id)}"
            )
        for sample in self.samples:
            cached = self.cache_by_id[sample["sample_id"]]
            if str(cached.get("category")) != str(sample["category"]):
                raise ValueError(
                    f"Category mismatch for sample_id={sample['sample_id']!r}: "
                    f"manifest={sample['category']!r} cache={cached.get('category')!r}"
                )
            for key in ("input_path", "target_path", "index_path"):
                cached_value = cached.get(key)
                if cached_value is not None and str(cached_value) != str(sample[key]):
                    raise ValueError(
                        f"Path mismatch for sample_id={sample['sample_id']!r}, field={key!r}: "
                        f"manifest={sample[key]!r} cache={cached_value!r}"
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        sample_id = str(sample["sample_id"])
        category = str(sample["category"])
        if category not in self.category_to_id:
            available = sorted(self.category_to_id)
            raise KeyError(
                f"Category {category!r} missing from category_to_id for sample_id={sample_id!r} "
                f"in manifest={self.manifest_path}. Available categories: {available}"
            )
        cached = self.cache_by_id.get(sample_id)
        if cached is None:
            raise KeyError(
                f"Missing plan cache entry for sample_id={sample_id!r} "
                f"in cache manifest={self.cache_meta.get('manifest', 'unknown_manifest')}"
            )
        y20 = cached.get("y20")
        if not isinstance(y20, list):
            raise ValueError(f"Invalid y20 in cache for sample_id={sample_id!r}")
        fg_stats = self.category_fg_stats.get(category)
        occ_stats = self.category_occ_stats.get(category)
        if not isinstance(fg_stats, dict):
            raise ValueError(f"Missing category_fg_stats for category={category!r} in cache {self.cache_meta.get('manifest')}")
        if not isinstance(occ_stats, dict):
            raise ValueError(f"Missing category_occ_stats for category={category!r} in cache {self.cache_meta.get('manifest')}")
        required_fg = ["count", "mean", "std", "q01", "q05", "q10", "q50", "q90", "q95", "q99", "valid_low", "valid_high"]
        required_occ = ["valid_o5_min_cells", "valid_o5_max_cells", "valid_o10_min_cells", "valid_o10_max_cells"]
        missing_fg = [key for key in required_fg if key not in fg_stats]
        missing_occ = [key for key in required_occ if key not in occ_stats]
        if missing_fg:
            raise ValueError(f"Missing fg stats keys for category={category!r}: {missing_fg}")
        if missing_occ:
            raise ValueError(f"Missing occ stats keys for category={category!r}: {missing_occ}")
        fg_valid_low = float(fg_stats["valid_low"])
        fg_valid_high = float(fg_stats["valid_high"])
        o5_valid_low_ratio = float(occ_stats["valid_o5_min_cells"]) / 25.0
        o5_valid_high_ratio = float(occ_stats["valid_o5_max_cells"]) / 25.0
        o10_valid_low_ratio = float(occ_stats["valid_o10_min_cells"]) / 100.0
        o10_valid_high_ratio = float(occ_stats["valid_o10_max_cells"]) / 100.0
        return {
            "sample_id": sample_id,
            "category": category,
            "category_id": int(self.category_to_id[category]),
            "y20": getattr(torch, "tensor")(y20, dtype=getattr(torch, "long")),
            "z": getattr(torch, "tensor")(int(cached.get("local_z", cached["z"])), dtype=getattr(torch, "long")),
            "local_z": getattr(torch, "tensor")(int(cached.get("local_z", cached["z"])), dtype=getattr(torch, "long")),
            "num_modes_for_category": getattr(torch, "tensor")(int(cached.get("num_modes_for_category", self.max_num_modes_per_category)), dtype=getattr(torch, "long")),
            "mode_mask": getattr(torch, "tensor")(
                [1 if mode_index < int(cached.get("num_modes_for_category", self.max_num_modes_per_category)) else 0 for mode_index in range(self.max_num_modes_per_category)],
                dtype=getattr(torch, "bool"),
            ),
            "c5": getattr(torch, "tensor")(cached["c5"], dtype=getattr(torch, "long")),
            "o5": getattr(torch, "tensor")(cached["o5"], dtype=getattr(torch, "float32")),
            "c10": getattr(torch, "tensor")(cached["c10"], dtype=getattr(torch, "long")),
            "o10": getattr(torch, "tensor")(cached["o10"], dtype=getattr(torch, "float32")),
            "r17": getattr(torch, "tensor")(cached["r17"], dtype=getattr(torch, "float32")),
            "fg_ratio": getattr(torch, "tensor")(float(cached["fg_ratio"]), dtype=getattr(torch, "float32")),
            "row_projection": getattr(torch, "tensor")(cached["row_projection"], dtype=getattr(torch, "float32")),
            "col_projection": getattr(torch, "tensor")(cached["col_projection"], dtype=getattr(torch, "float32")),
            "grammar_signature": getattr(torch, "tensor")(cached["grammar_signature"], dtype=getattr(torch, "float32")),
            "adjacency_signature": getattr(torch, "tensor")(cached["adjacency_signature"], dtype=getattr(torch, "float32")),
            "fg_valid_low": getattr(torch, "tensor")(fg_valid_low, dtype=getattr(torch, "float32")),
            "fg_valid_high": getattr(torch, "tensor")(fg_valid_high, dtype=getattr(torch, "float32")),
            "o5_valid_low_ratio": getattr(torch, "tensor")(o5_valid_low_ratio, dtype=getattr(torch, "float32")),
            "o5_valid_high_ratio": getattr(torch, "tensor")(o5_valid_high_ratio, dtype=getattr(torch, "float32")),
            "o10_valid_low_ratio": getattr(torch, "tensor")(o10_valid_low_ratio, dtype=getattr(torch, "float32")),
            "o10_valid_high_ratio": getattr(torch, "tensor")(o10_valid_high_ratio, dtype=getattr(torch, "float32")),
            "metadata": cached,
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    torch, _ = _require_torch()
    return {
        "sample_ids": [str(sample["sample_id"]) for sample in batch],
        "categories": [str(sample["category"]) for sample in batch],
        "category_ids": getattr(torch, "tensor")([int(sample["category_id"]) for sample in batch], dtype=getattr(torch, "long")),
        "y20": getattr(torch, "stack")([sample["y20"] for sample in batch]),
        "z": getattr(torch, "stack")([sample["z"] for sample in batch]),
        "local_z": getattr(torch, "stack")([sample["local_z"] for sample in batch]),
        "num_modes_for_category": getattr(torch, "stack")([sample["num_modes_for_category"] for sample in batch]),
        "mode_mask": getattr(torch, "stack")([sample["mode_mask"] for sample in batch]),
        "c5": getattr(torch, "stack")([sample["c5"] for sample in batch]),
        "o5": getattr(torch, "stack")([sample["o5"] for sample in batch]),
        "c10": getattr(torch, "stack")([sample["c10"] for sample in batch]),
        "o10": getattr(torch, "stack")([sample["o10"] for sample in batch]),
        "r17": getattr(torch, "stack")([sample["r17"] for sample in batch]),
        "fg_ratio": getattr(torch, "stack")([sample["fg_ratio"] for sample in batch]),
        "row_projection": getattr(torch, "stack")([sample["row_projection"] for sample in batch]),
        "col_projection": getattr(torch, "stack")([sample["col_projection"] for sample in batch]),
        "grammar_signature": getattr(torch, "stack")([sample["grammar_signature"] for sample in batch]),
        "adjacency_signature": getattr(torch, "stack")([sample["adjacency_signature"] for sample in batch]),
        "fg_valid_low": getattr(torch, "stack")([sample["fg_valid_low"] for sample in batch]),
        "fg_valid_high": getattr(torch, "stack")([sample["fg_valid_high"] for sample in batch]),
        "o5_valid_low_ratio": getattr(torch, "stack")([sample["o5_valid_low_ratio"] for sample in batch]),
        "o5_valid_high_ratio": getattr(torch, "stack")([sample["o5_valid_high_ratio"] for sample in batch]),
        "o10_valid_low_ratio": getattr(torch, "stack")([sample["o10_valid_low_ratio"] for sample in batch]),
        "o10_valid_high_ratio": getattr(torch, "stack")([sample["o10_valid_high_ratio"] for sample in batch]),
        "metadata": [sample["metadata"] for sample in batch],
    }


def build_dataloader(
    manifest_path: str | Path,
    palette_path: str | Path | None,
    plan_cache_path: str | Path,
    batch_size: int,
    shuffle: bool,
    category_to_id: dict[str, int] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[object, LatentPlanDataset]:
    _, data = _require_torch()
    dataset = LatentPlanDataset(manifest_path, palette_path=palette_path, plan_cache_path=plan_cache_path, category_to_id=category_to_id)
    loader = getattr(data, "DataLoader")(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers if num_workers > 0 else False),
    )
    return loader, dataset


__all__ = [
    "LatentPlanDataset",
    "build_dataloader",
    "collate_batch",
    "load_manifest",
]
