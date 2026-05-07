from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .dataset import (
    ColorVocabulary,
    SimulationTopologyDataset,
    SegmentationTarget,
    build_topk_color_vocabulary,
    crop_image,
    downsample_color_grid,
    infer_active_crop,
    load_parser_manifest,
    load_rgb_image,
    resize_image,
    color_grid_to_class_grid,
    write_grid_json,
    write_vocabulary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline cached teacher targets for dataset2.")
    parser.add_argument("--manifest", type=Path, required=True, help="Source manifest JSONL")
    parser.add_argument("--output-root", type=Path, required=True, help="Cache output root")
    parser.add_argument("--image-size", type=int, nargs=2, default=(160, 160), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--grid-size", type=int, nargs=2, default=(20, 20), metavar=("ROWS", "COLS"))
    parser.add_argument("--top-k-colors", type=int, default=2)
    return parser


def _print_progress(current: int, total: int) -> None:
    width = 30
    ratio = 0.0 if total == 0 else current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}", end="", flush=True)


def _build_samples_from_manifest(manifest_path: Path) -> tuple[list[SegmentationTarget], Path]:
    raw_samples = load_parser_manifest(manifest_path)
    root = SimulationTopologyDataset._infer_root(manifest_path)
    samples = [
        SegmentationTarget(
            sample_id=sample["sample_id"],
            category=sample["category"],
            image_path=(root / sample["image_path"]).resolve(),
            target_path=(root / sample["target_path"]).resolve(),
        )
        for sample in raw_samples
    ]
    return samples, root


def _build_vocabulary_with_progress(
    samples: list[SegmentationTarget],
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    top_k: int,
) -> ColorVocabulary:
    from collections import Counter

    counter: Counter[tuple[int, int, int]] = Counter()
    total = len(samples)
    print(f"[phase 1/2] building vocabulary from {total} samples")
    for index, sample in enumerate(samples, start=1):
        target_image = load_rgb_image(sample.target_path)
        crop_box = infer_active_crop(target_image)
        resized = resize_image(crop_image(target_image, crop_box), image_size, nearest=True)
        color_grid = downsample_color_grid(resized, grid_size)
        for row in color_grid:
            counter.update(row)
        _print_progress(index, total)
    print()
    top_colors = tuple(color for color, _ in counter.most_common(top_k))
    class_names = tuple([f"color_{index}" for index in range(len(top_colors))] + ["other"])
    return ColorVocabulary(
        top_colors=top_colors,
        color_to_class={color: idx for idx, color in enumerate(top_colors)},
        class_names=class_names,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    grid_size = (int(args.grid_size[0]), int(args.grid_size[1]))
    samples, root = _build_samples_from_manifest(args.manifest)
    vocabulary = _build_vocabulary_with_progress(samples, image_size=image_size, grid_size=grid_size, top_k=args.top_k_colors)
    dataset = SimulationTopologyDataset(
        args.manifest,
        root=root,
        image_size=image_size,
        grid_size=grid_size,
        vocabulary=vocabulary,
        top_k_colors=args.top_k_colors,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_vocabulary(args.output_root / "vocabulary.json", dataset.vocabulary)

    cached_rows: list[dict[str, object]] = []
    total = len(dataset.samples)
    print(f"[phase 2/2] writing cache for {total} samples")
    for index, sample in enumerate(dataset.samples, start=1):
        source_image = load_rgb_image(sample.image_path)
        source_target = load_rgb_image(sample.target_path)
        crop_box = infer_active_crop(source_target)
        image = resize_image(crop_image(source_image, crop_box), dataset.image_size, nearest=False).convert("L")
        target_image = resize_image(crop_image(source_target, crop_box), dataset.image_size, nearest=True)
        color_grid = downsample_color_grid(target_image, dataset.grid_size)
        class_grid = color_grid_to_class_grid(color_grid, dataset.vocabulary)

        sample_dir = args.output_root / sample.sample_id.replace("/", "__")
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = sample_dir / "input.png"
        target_path = sample_dir / "target_grid.json"
        image.save(image_path)
        write_grid_json(target_path, class_grid)

        cached_rows.append(
            {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "image_path": str(image_path.relative_to(args.output_root)).replace("\\", "/"),
                "target_path": str(target_path.relative_to(args.output_root)).replace("\\", "/"),
            }
        )
        _print_progress(index, total)

    print()
    manifest_path = args.output_root / "manifest.jsonl"
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in cached_rows)
    if text:
        text += "\n"
    manifest_path.write_text(text, encoding="utf-8")
    print(json.dumps({"cache_root": str(args.output_root), "manifest": str(manifest_path), "samples": len(cached_rows)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
