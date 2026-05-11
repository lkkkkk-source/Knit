from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

from .utils import IGNORE_INDEX, EXPECTED_DESCRIPTOR_DIM, require_centroid_sketch_fields, require_ignore_index, resolve_canonical_mode, validate_foreground_labels


class ForegroundSample(TypedDict):
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
        raise ImportError("PyTorch is required for struct_foreground_v1 dataset.") from error
    return torch, data


def load_manifest(path: str | Path) -> list[ForegroundSample]:
    rows: list[ForegroundSample] = []
    path = Path(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        row = {}
        for key in ["sample_id", "category", "input_path", "target_path", "index_path"]:
            value = payload.get(key)
            if not isinstance(value, str):
                raise ValueError(f"Missing {key!r} in manifest row: {payload!r}")
            row[key] = value
        rows.append(cast(ForegroundSample, row))
    return rows


class ForegroundDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        cache_path: str | Path,
        category_to_id: dict[str, int] | None = None,
        *,
        exclude_unseen_categories: bool = False,
    ) -> None:
        torch, _ = _require_torch()
        self.manifest_path = Path(manifest_path)
        self.samples = load_manifest(self.manifest_path)
        self.cache_payload = getattr(torch, "load")(Path(cache_path), map_location="cpu")
        self.cache_by_id = {entry["sample_id"]: entry for entry in self.cache_payload["items"]}
        self.exclude_unseen_categories = bool(exclude_unseen_categories)
        self.skipped_unseen_count = 0
        self.skipped_unseen_categories: list[str] = []
        categories = sorted({sample["category"] for sample in self.samples})
        self.category_to_id = category_to_id or {category: index for index, category in enumerate(categories)}
        planner_cf = self.cache_payload.get("config", {}).get("planner", {})
        data_cf = self.cache_payload.get("config", {}).get("data", {})
        self.canonical_mode = resolve_canonical_mode(data_cf)
        self.ignore_index = require_ignore_index(data_cf)
        self.max_num_modes_per_category = int(planner_cf.get("max_num_modes_per_category", planner_cf.get("num_modes_per_category", 16)))
        self._validate_alignment()
        require_centroid_sketch_fields(self.cache_payload, context="Foreground dataset cache")
        if self.exclude_unseen_categories:
            filtered = []
            skipped_categories = set()
            for sample in self.samples:
                cached = self.cache_by_id[sample["sample_id"]]
                if bool(cached.get("is_unseen_category", False)):
                    self.skipped_unseen_count += 1
                    skipped_categories.add(str(sample["category"]))
                    continue
                filtered.append(sample)
            self.samples = filtered
            self.skipped_unseen_categories = sorted(skipped_categories)

    def _validate_alignment(self) -> None:
        manifest_ids = {sample["sample_id"] for sample in self.samples}
        cache_ids = set(self.cache_by_id)
        if manifest_ids != cache_ids:
            missing_in_cache = sorted(manifest_ids - cache_ids)
            missing_in_manifest = sorted(cache_ids - manifest_ids)
            raise ValueError(
                f"Manifest/cache sample_id mismatch in foreground dataset: "
                f"missing_in_cache={missing_in_cache[:5]} missing_in_manifest={missing_in_manifest[:5]}"
            )
        for sample in self.samples:
            cached = self.cache_by_id[sample["sample_id"]]
            if cached["category"] != sample["category"]:
                raise ValueError(f"Category mismatch for sample_id={sample['sample_id']}")
            for key in ("input_path", "target_path", "index_path"):
                if str(cached.get(key)) != str(sample[key]):
                    raise ValueError(f"Path mismatch for sample_id={sample['sample_id']} field={key}")
            validate_foreground_labels(cached["fg_y20"], cached["fg_mask20"], context=f"dataset[{sample['sample_id']}]")
            if cached.get("canonical_mode", self.canonical_mode) != self.canonical_mode:
                raise ValueError(
                    f"Canonical mode mismatch for sample_id={sample['sample_id']}: "
                    f"cache has {cached.get('canonical_mode')!r}, dataset expects {self.canonical_mode!r}"
                )
            descriptor = cached.get("descriptor")
            if not isinstance(descriptor, list) or len(descriptor) != EXPECTED_DESCRIPTOR_DIM:
                raise ValueError(
                    f"Descriptor mismatch for sample_id={sample['sample_id']}: expected dim {EXPECTED_DESCRIPTOR_DIM}, got {len(descriptor) if isinstance(descriptor, list) else type(descriptor)}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        cached = self.cache_by_id[sample["sample_id"]]
        category = sample["category"]
        if category not in self.category_to_id:
            raise KeyError(f"Category {category!r} missing from category_to_id.")
        if bool(cached.get("is_unseen_category", False)):
            raise ValueError(
                f"Unseen category sample cannot provide centroid sketch: category={category}, sample_id={sample['sample_id']}"
            )
        local_z = int(cached["local_z"])
        num_modes = int(cached["num_modes_for_category"])
        if num_modes <= 0 or num_modes > self.max_num_modes_per_category:
            raise ValueError(
                f"Invalid num_modes_for_category for sample_id={sample['sample_id']}: {num_modes} not in [1, {self.max_num_modes_per_category}]"
            )
        if local_z < 0 or local_z >= num_modes:
            raise ValueError(f"Invalid local_z for sample_id={sample['sample_id']}: local_z={local_z} num_modes={num_modes}")
        mode_mask = [1 if mode_index < num_modes else 0 for mode_index in range(self.max_num_modes_per_category)]
        centroid = self.cache_payload["centroid_sketch_by_category"].get(category, {}).get(local_z, {})
        if not centroid:
            raise ValueError(f"Missing centroid sketch for sample_id={sample['sample_id']} category={category!r} local_z={local_z}.")
        if "centroid_fg_mask_prob" not in centroid or "centroid_fg_mask" not in centroid or "centroid_label_prob_16" not in centroid:
            raise ValueError(
                f"Centroid sketch for sample_id={sample['sample_id']} category={category!r} local_z={local_z} "
                "is missing centroid_fg_mask_prob / centroid_fg_mask / centroid_label_prob_16. Please rebuild the cache."
            )
        centroid_fg_mask_prob = centroid.get("centroid_fg_mask_prob")
        centroid_fg_mask_bin = centroid.get("centroid_fg_mask")
        centroid_label_prob_16 = centroid.get("centroid_label_prob_16")
        if hasattr(centroid_fg_mask_prob, "tolist"):
            centroid_fg_mask_prob = centroid_fg_mask_prob.tolist()
        if hasattr(centroid_fg_mask_bin, "tolist"):
            centroid_fg_mask_bin = centroid_fg_mask_bin.tolist()
        if hasattr(centroid_label_prob_16, "tolist"):
            centroid_label_prob_16 = centroid_label_prob_16.tolist()
        if not isinstance(centroid_fg_mask_prob, list) or len(centroid_fg_mask_prob) != 1 or not isinstance(centroid_fg_mask_prob[0], list) or len(centroid_fg_mask_prob[0]) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in centroid_fg_mask_prob[0]):
            raise ValueError(f"centroid_fg_mask_prob for sample_id={sample['sample_id']} must have shape [1,20,20].")
        if not isinstance(centroid_fg_mask_bin, list) or len(centroid_fg_mask_bin) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in centroid_fg_mask_bin):
            raise ValueError(f"centroid_fg_mask for sample_id={sample['sample_id']} must have shape [20,20].")
        if not isinstance(centroid_label_prob_16, list) or len(centroid_label_prob_16) != 16 or any(not isinstance(channel, list) or len(channel) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in channel) for channel in centroid_label_prob_16):
            raise ValueError(f"centroid_label_prob_16 for sample_id={sample['sample_id']} must have shape [16,20,20].")
        if sum(int(value) for row in centroid_fg_mask_bin for value in row) <= 0:
            raise ValueError(f"centroid_fg_mask for sample_id={sample['sample_id']} category={category!r} local_z={local_z} is all-zero.")
        return {
            "sample_id": str(sample["sample_id"]),
            "category": str(category),
            "category_id": int(self.category_to_id[category]),
            "local_z": getattr(torch, "tensor")(local_z, dtype=getattr(torch, "long")),
            "mode_mask": getattr(torch, "tensor")(mode_mask, dtype=getattr(torch, "bool")),
            "fg_y20": getattr(torch, "tensor")(cached["fg_y20"], dtype=getattr(torch, "long")),
            "fg_mask20": getattr(torch, "tensor")(cached["fg_mask20"], dtype=getattr(torch, "float32")),
            "bbox_stats": getattr(torch, "tensor")(cached["bbox_stats"], dtype=getattr(torch, "float32")),
            "label_hist_16": getattr(torch, "tensor")(cached["label_hist_16"], dtype=getattr(torch, "float32")),
            "row_projection": getattr(torch, "tensor")(cached["row_projection"], dtype=getattr(torch, "float32")),
            "col_projection": getattr(torch, "tensor")(cached["col_projection"], dtype=getattr(torch, "float32")),
            "grammar_signature": getattr(torch, "tensor")(cached["grammar_signature"], dtype=getattr(torch, "float32")),
            "adjacency_signature": getattr(torch, "tensor")(cached["adjacency_signature"], dtype=getattr(torch, "float32")),
            "descriptor": getattr(torch, "tensor")(cached["descriptor"], dtype=getattr(torch, "float32")),
            "fg_area": getattr(torch, "tensor")(float(cached["fg_area"]), dtype=getattr(torch, "float32")),
            "centroid_fg_mask_prob": getattr(torch, "tensor")(centroid_fg_mask_prob, dtype=getattr(torch, "float32")),
            "centroid_fg_mask_bin": getattr(torch, "tensor")(centroid_fg_mask_bin, dtype=getattr(torch, "float32")).unsqueeze(0),
            "centroid_label_prob_16": getattr(torch, "tensor")(centroid_label_prob_16, dtype=getattr(torch, "float32")),
            "centroid_label_hist": getattr(torch, "tensor")(centroid.get("centroid_label_hist", [0.0] * 16), dtype=getattr(torch, "float32")),
            "centroid_row_projection": getattr(torch, "tensor")(centroid.get("centroid_row_projection", [0.0] * 20), dtype=getattr(torch, "float32")),
            "centroid_col_projection": getattr(torch, "tensor")(centroid.get("centroid_col_projection", [0.0] * 20), dtype=getattr(torch, "float32")),
            "centroid_adjacency": getattr(torch, "tensor")(centroid.get("centroid_adjacency", [0.0] * 256), dtype=getattr(torch, "float32")),
            "centroid_transition_stats": getattr(torch, "tensor")(centroid.get("centroid_transition_stats", [0.0] * 6), dtype=getattr(torch, "float32")),
            "centroid_bbox_stats": getattr(torch, "tensor")(centroid.get("centroid_bbox_stats", [0.0] * 10), dtype=getattr(torch, "float32")),
            "original_y20": getattr(torch, "tensor")(cached["original_y20"], dtype=getattr(torch, "long")),
            "canonical_mode": str(self.canonical_mode),
            "metadata": cached,
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    torch, _ = _require_torch()
    return {
        "sample_ids": [sample["sample_id"] for sample in batch],
        "categories": [sample["category"] for sample in batch],
        "category_ids": getattr(torch, "tensor")([int(sample["category_id"]) for sample in batch], dtype=getattr(torch, "long")),
        "local_z": getattr(torch, "stack")([sample["local_z"] for sample in batch]),
        "mode_mask": getattr(torch, "stack")([sample["mode_mask"] for sample in batch]),
        "fg_y20": getattr(torch, "stack")([sample["fg_y20"] for sample in batch]),
        "fg_mask20": getattr(torch, "stack")([sample["fg_mask20"] for sample in batch]),
        "bbox_stats": getattr(torch, "stack")([sample["bbox_stats"] for sample in batch]),
        "label_hist_16": getattr(torch, "stack")([sample["label_hist_16"] for sample in batch]),
        "row_projection": getattr(torch, "stack")([sample["row_projection"] for sample in batch]),
        "col_projection": getattr(torch, "stack")([sample["col_projection"] for sample in batch]),
        "grammar_signature": getattr(torch, "stack")([sample["grammar_signature"] for sample in batch]),
        "adjacency_signature": getattr(torch, "stack")([sample["adjacency_signature"] for sample in batch]),
        "descriptor": getattr(torch, "stack")([sample["descriptor"] for sample in batch]),
        "fg_area": getattr(torch, "stack")([sample["fg_area"] for sample in batch]),
        "centroid_fg_mask_prob": getattr(torch, "stack")([sample["centroid_fg_mask_prob"] for sample in batch]),
        "centroid_fg_mask_bin": getattr(torch, "stack")([sample["centroid_fg_mask_bin"] for sample in batch]),
        "centroid_label_prob_16": getattr(torch, "stack")([sample["centroid_label_prob_16"] for sample in batch]),
        "centroid_label_hist": getattr(torch, "stack")([sample["centroid_label_hist"] for sample in batch]),
        "centroid_row_projection": getattr(torch, "stack")([sample["centroid_row_projection"] for sample in batch]),
        "centroid_col_projection": getattr(torch, "stack")([sample["centroid_col_projection"] for sample in batch]),
        "centroid_adjacency": getattr(torch, "stack")([sample["centroid_adjacency"] for sample in batch]),
        "centroid_transition_stats": getattr(torch, "stack")([sample["centroid_transition_stats"] for sample in batch]),
        "centroid_bbox_stats": getattr(torch, "stack")([sample["centroid_bbox_stats"] for sample in batch]),
        "original_y20": getattr(torch, "stack")([sample["original_y20"] for sample in batch]),
        "canonical_mode": batch[0]["canonical_mode"] if batch else "full_masked",
        "metadata": [sample["metadata"] for sample in batch],
    }


def build_dataloader(
    manifest_path: str | Path,
    cache_path: str | Path,
    batch_size: int,
    shuffle: bool,
    category_to_id: dict[str, int] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    exclude_unseen_categories: bool = False,
) -> tuple[object, ForegroundDataset]:
    _, data = _require_torch()
    dataset = ForegroundDataset(
        manifest_path,
        cache_path=cache_path,
        category_to_id=category_to_id,
        exclude_unseen_categories=exclude_unseen_categories,
    )
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
