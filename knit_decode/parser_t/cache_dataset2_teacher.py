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
    mask_to_image,
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
    parser.add_argument("--shard-size", type=int, default=512, help="Number of samples per cached shard")
    parser.add_argument("--preview-count", type=int, default=8, help="How many preview PNG pairs to keep for inspection")
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
        resized = resize_image(target_image, image_size, nearest=True)
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
    torch, _ = dataset._require_torch() if False else __import__("importlib").import_module("torch"), None
    shard_images: list[object] = []
    shard_targets: list[object] = []
    shard_sample_ids: list[str] = []
    shard_index = 0
    class_pixel_counts = [0 for _ in range(dataset.num_classes)]
    preview_dir = args.output_root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(dataset.samples, start=1):
        source_image = load_rgb_image(sample.image_path)
        source_target = load_rgb_image(sample.target_path)
        image = resize_image(source_image, dataset.image_size, nearest=False).convert("L")
        target_image = resize_image(source_target, dataset.image_size, nearest=True)
        color_grid = downsample_color_grid(target_image, dataset.grid_size)
        class_grid = color_grid_to_class_grid(color_grid, dataset.vocabulary)
        image_tensor = torch.tensor(list(image.getdata()), dtype=torch.uint8).reshape(1, image.height, image.width)
        target_tensor = torch.tensor(class_grid, dtype=torch.long)
        shard_images.append(image_tensor)
        shard_targets.append(target_tensor)
        shard_sample_ids.append(sample.sample_id)
        for row in class_grid:
            for class_id in row:
                class_pixel_counts[class_id] += 1

        if index <= args.preview_count:
            sample_dir = preview_dir / sample.sample_id.replace("/", "__")
            sample_dir.mkdir(parents=True, exist_ok=True)
            image.save(sample_dir / "input.png")
            mask_to_image(class_grid, dataset.vocabulary).save(sample_dir / "target.png")

        if len(shard_images) >= args.shard_size or index == total:
            shard_path = args.output_root / f"train_shard_{shard_index:04d}.pt"
            torch.save(
                {
                    "images": torch.stack(shard_images),
                    "targets": torch.stack(shard_targets),
                    "sample_ids": shard_sample_ids,
                },
                shard_path,
            )
            for item_offset, sample_id in enumerate(shard_sample_ids):
                cached_rows.append(
                    {
                        "sample_id": sample_id,
                        "category": sample_id.split("/", 1)[0] if "/" in sample_id else sample_id.split("_", 1)[0],
                        "shard_path": str(shard_path.relative_to(args.output_root)).replace("\\", "/"),
                        "item_index": item_offset,
                    }
                )
            shard_images = []
            shard_targets = []
            shard_sample_ids = []
            shard_index += 1
        _print_progress(index, total)

    print()
    manifest_path = args.output_root / "manifest.jsonl"
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in cached_rows)
    if text:
        text += "\n"
    manifest_path.write_text(text, encoding="utf-8")
    metadata = {
        "cache_root": str(args.output_root),
        "manifest": str(manifest_path),
        "samples": len(cached_rows),
        "num_shards": shard_index,
        "class_pixel_counts": class_pixel_counts,
        "top_k_colors": args.top_k_colors,
        "image_size": list(image_size),
        "grid_size": list(grid_size),
    }
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
