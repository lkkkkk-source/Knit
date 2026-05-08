from __future__ import annotations

from pathlib import Path

from knit_decode.inverse_residual_v1.dataset import (
    ResidualJointDataset as _ResidualJointDataset,
    build_dataloader as _build_dataloader,
    collate_batch as _collate_batch,
    load_manifest,
)


class InverseFullDataset(_ResidualJointDataset):
    pass


def build_dataloader(
    manifest_path: str | Path,
    palette_path: str | Path,
    batch_size: int,
    shuffle: bool,
    image_size: tuple[int, int] = (160, 160),
    transfer_root: str | Path | None = None,
    use_best_crop: bool = False,
    augment_scale: float = 0.0,
    mean_value: float = 0.5,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[object, InverseFullDataset]:
    dataloader, dataset = _build_dataloader(
        manifest_path=manifest_path,
        palette_path=palette_path,
        batch_size=batch_size,
        shuffle=shuffle,
        image_size=image_size,
        transfer_root=transfer_root,
        use_best_crop=use_best_crop,
        augment_scale=augment_scale,
        mean_value=mean_value,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return dataloader, dataset  # type: ignore[return-value]


collate_batch = _collate_batch
