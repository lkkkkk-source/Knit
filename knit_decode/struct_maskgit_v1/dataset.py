from __future__ import annotations

from pathlib import Path

from knit_decode.struct_ar_v1.dataset import (
    NUM_CLASSES,
    StructureSampleDataset,
    build_dataloader as build_stage_dataset_dataloader,
    collate_batch,
    compute_class_counts,
    load_manifest,
    read_palette_mapping,
)


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
    return build_stage_dataset_dataloader(
        manifest_path,
        palette_path=palette_path,
        batch_size=batch_size,
        shuffle=shuffle,
        category_to_id=category_to_id,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


__all__ = [
    "NUM_CLASSES",
    "StructureSampleDataset",
    "build_dataloader",
    "collate_batch",
    "compute_class_counts",
    "load_manifest",
    "read_palette_mapping",
]
