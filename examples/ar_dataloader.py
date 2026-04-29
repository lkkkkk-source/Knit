from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from knit_decode.ar_dataset import build_ar_dataloader


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect AR-export batches with a PyTorch DataLoader")
    _ = parser.add_argument("export_root", type=Path, help="Output directory containing ar_manifest.jsonl and ar_vocab.json")
    _ = parser.add_argument("--batch-size", type=int, default=2, help="Batch size for the demo dataloader")
    args = parser.parse_args()
    export_root = cast(Path, args.export_root)
    batch_size = cast(int, args.batch_size)

    dataloader = cast(Iterable[dict[str, object]], build_ar_dataloader(export_root, batch_size=batch_size, shuffle=False))
    first_batch = next(iter(dataloader))
    input_shape = getattr(first_batch["input_ids"], "shape", None)
    target_shape = getattr(first_batch["target_ids"], "shape", None)
    grid_shape = getattr(first_batch["grid_ids"], "shape", None)
    attention_shape = getattr(first_batch["attention_mask"], "shape", None)
    grid_mask_shape = getattr(first_batch["grid_mask"], "shape", None)

    print(f"sample_ids: {first_batch['sample_ids']}")
    print(f"input_ids shape: {input_shape}")
    print(f"target_ids shape: {target_shape}")
    print(f"grid_ids shape: {grid_shape}")
    print(f"attention_mask shape: {attention_shape}")
    print(f"grid_mask shape: {grid_mask_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
