from __future__ import annotations

import json
from pathlib import Path


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        data = importlib.import_module("torch.utils.data")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_latentplan_v1 dataset. Install with `pip install -e .[train]`.") from error
    return torch, data


class LatentPlanDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        palette_path: str | Path,
        plan_cache_path: str | Path,
        category_to_id: dict[str, int] | None = None,
    ) -> None:
        from knit_decode.struct_ar_v1.dataset import StructureSampleDataset

        self.structure = StructureSampleDataset(manifest_path, palette_path=palette_path, category_to_id=category_to_id)
        self.category_to_id = self.structure.category_to_id
        self.samples = self.structure.samples
        self.num_classes = self.structure.num_classes
        torch, _ = _require_torch()
        cache_payload = getattr(torch, "load")(Path(plan_cache_path), map_location="cpu")
        self.cache_meta = cache_payload["meta"]
        self.cache_by_id = {entry["sample_id"]: entry for entry in cache_payload["items"]}

    def __len__(self) -> int:
        return len(self.structure)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.structure[index]
        sample_id = str(sample["sample_id"])
        cached = self.cache_by_id.get(sample_id)
        if cached is None:
            raise KeyError(f"Missing plan cache entry for sample_id={sample_id!r}")
        return {
            "sample_id": sample_id,
            "category": str(sample["category"]),
            "category_id": int(sample["category_id"]),
            "y20": sample["grid20"],
            "z": getattr(torch, "tensor")(int(cached["z"]), dtype=getattr(torch, "long")),
            "c5": getattr(torch, "tensor")(cached["c5"], dtype=getattr(torch, "long")),
            "o5": getattr(torch, "tensor")(cached["o5"], dtype=getattr(torch, "float32")),
            "r17": getattr(torch, "tensor")(cached["r17"], dtype=getattr(torch, "float32")),
            "fg_ratio": getattr(torch, "tensor")(float(cached["fg_ratio"]), dtype=getattr(torch, "float32")),
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
        "c5": getattr(torch, "stack")([sample["c5"] for sample in batch]),
        "o5": getattr(torch, "stack")([sample["o5"] for sample in batch]),
        "r17": getattr(torch, "stack")([sample["r17"] for sample in batch]),
        "fg_ratio": getattr(torch, "stack")([sample["fg_ratio"] for sample in batch]),
        "metadata": [sample["metadata"] for sample in batch],
    }


def build_dataloader(
    manifest_path: str | Path,
    palette_path: str | Path,
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
]
