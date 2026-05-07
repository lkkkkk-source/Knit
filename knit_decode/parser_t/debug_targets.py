from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import (
    SimulationTopologyDataset,
    crop_image,
    downsample_semantic_grid,
    infer_active_crop,
    load_rgb_image,
    mask_to_image,
    resize_image,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug parser target generation for a single sample.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSONL file")
    parser.add_argument("--sample-id", type=str, required=True, help="Sample id, for example `Tuck/040`")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write debug visualizations")
    parser.add_argument("--image-size", type=int, nargs=2, default=(160, 160), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--grid-size", type=int, nargs=2, default=(20, 20), metavar=("ROWS", "COLS"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset = SimulationTopologyDataset(
        args.manifest,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        grid_size=(int(args.grid_size[0]), int(args.grid_size[1])),
    )
    sample = next((item for item in dataset.samples if item.sample_id == args.sample_id), None)
    if sample is None:
        raise ValueError(f"Sample {args.sample_id!r} was not found in {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_image = load_rgb_image(sample.target_path)
    crop_box = infer_active_crop(target_image)
    cropped = crop_image(target_image, crop_box)
    resized = resize_image(cropped, dataset.image_size, nearest=True)
    semantic_grid = downsample_semantic_grid(resized, dataset.grid_size, dataset.color_to_class)

    target_image.save(args.output_dir / "target_original.png")
    cropped.save(args.output_dir / "target_cropped.png")
    resized.save(args.output_dir / "target_resized.png")
    mask_to_image(semantic_grid).save(args.output_dir / "target_semantic_grid.png")
    (args.output_dir / "target_semantic_grid.txt").write_text(
        "\n".join(" ".join(str(value) for value in row) for row in semantic_grid) + "\n",
        encoding="utf-8",
    )
    print(f"saved debug outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
