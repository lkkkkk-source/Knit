from __future__ import annotations

from pathlib import Path

from knit_decode.parser_t_inverse.dataset import (
    NUM_CLASSES,
    ParserInverseDataset,
    compute_class_counts,
    load_manifest,
    read_palette_mapping,
)


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        data = importlib.import_module("torch.utils.data")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_ar_v1 training. Install with `pip install -e .[train]`.") from error
    return torch, data


def _downsample_grid_nearest(grid: list[list[int]], size: int) -> list[list[int]]:
    src_h = len(grid)
    src_w = len(grid[0]) if src_h else 0
    if src_h != src_w:
        raise ValueError("Expected square grid")
    out: list[list[int]] = []
    for y in range(size):
        row: list[int] = []
        y0 = (y * src_h) // size
        for x in range(size):
            x0 = (x * src_w) // size
            row.append(int(grid[y0][x0]))
        out.append(row)
    return out


class StructureSampleDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        palette_path: str | Path,
        category_to_id: dict[str, int] | None = None,
    ) -> None:
        self.base = ParserInverseDataset(manifest_path, palette_path=palette_path, image_size=(160, 160))
        self.samples = self.base.samples
        self.root = self.base.root
        self.class_names = list(self.base.class_names)
        self.num_classes = NUM_CLASSES
        categories = sorted({sample["category"] for sample in self.samples})
        self.category_to_id = category_to_id or {category: index for index, category in enumerate(categories)}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        item = self.base[index]
        grid20 = item["target"].tolist()
        grid10 = _downsample_grid_nearest(grid20, 10)
        grid5 = _downsample_grid_nearest(grid20, 5)
        sample = self.samples[index]
        return {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "category_id": self.category_to_id[sample["category"]],
            "grid5": getattr(torch, "tensor")(grid5, dtype=getattr(torch, "long")),
            "grid10": getattr(torch, "tensor")(grid10, dtype=getattr(torch, "long")),
            "grid20": item["target"],
            "count_vector": item["count_vector"],
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    torch, _ = _require_torch()
    return {
        "sample_ids": [str(sample["sample_id"]) for sample in batch],
        "categories": [str(sample["category"]) for sample in batch],
        "category_ids": getattr(torch, "tensor")([int(sample["category_id"]) for sample in batch], dtype=getattr(torch, "long")),
        "grid5": getattr(torch, "stack")([sample["grid5"] for sample in batch]),
        "grid10": getattr(torch, "stack")([sample["grid10"] for sample in batch]),
        "grid20": getattr(torch, "stack")([sample["grid20"] for sample in batch]),
        "count_vectors": getattr(torch, "stack")([sample["count_vector"] for sample in batch]),
    }


def build_dataloader(
    manifest_path: str | Path,
    palette_path: str | Path,
    batch_size: int,
    shuffle: bool,
    category_to_id: dict[str, int] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[object, StructureSampleDataset]:
    _, data = _require_torch()
    dataset = StructureSampleDataset(manifest_path, palette_path=palette_path, category_to_id=category_to_id)
    dataloader_cls = getattr(data, "DataLoader")
    worker_persistent = persistent_workers if num_workers > 0 else False
    return dataloader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=worker_persistent,
    ), dataset


__all__ = [
    "NUM_CLASSES",
    "StructureSampleDataset",
    "build_dataloader",
    "collate_batch",
    "compute_class_counts",
    "load_manifest",
    "read_palette_mapping",
]
