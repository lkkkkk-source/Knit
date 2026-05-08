from __future__ import annotations

from pathlib import Path

from knit_decode.refiner_v1.dataset import (
    PairedRefinerDataset as _BasePairedRefinerDataset,
    build_refiner_dataloader as _build_refiner_dataloader,
    collate_refiner_batch as _collate_refiner_batch,
    load_refiner_manifest,
)


class PairedInverseRefinerDataset(_BasePairedRefinerDataset):
    pass


def build_dataloader(
    manifest_path: str | Path,
    batch_size: int,
    shuffle: bool,
    image_size: tuple[int, int] = (160, 160),
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[object, PairedInverseRefinerDataset]:
    dataloader, dataset = _build_refiner_dataloader(
        manifest_path,
        batch_size=batch_size,
        shuffle=shuffle,
        image_size=image_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return dataloader, dataset  # type: ignore[return-value]


collate_batch = _collate_refiner_batch
