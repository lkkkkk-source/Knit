from __future__ import annotations

import argparse
import json
from pathlib import Path


RGB = tuple[int, int, int]
OFFICIAL_PALETTE: tuple[RGB, ...] = (
    (255, 0, 16),
    (43, 206, 72),
    (255, 255, 128),
    (94, 241, 242),
    (0, 129, 69),
    (0, 92, 49),
    (255, 0, 190),
    (194, 0, 136),
    (126, 0, 149),
    (96, 0, 112),
    (179, 179, 179),
    (128, 128, 128),
    (255, 230, 6),
    (255, 164, 4),
    (0, 164, 255),
    (0, 117, 220),
    (117, 59, 59),
)


def _load_bitmap(path: Path) -> object:
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError("Pillow is required for palette inference. Install with `pip install Pillow`.") from error
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def official_palette_mapping() -> dict[str, int]:
    return {f"{r},{g},{b}": index + 1 for index, (r, g, b) in enumerate(OFFICIAL_PALETTE)}


def infer_palette_mapping(manifest_path: str | Path, output_path: str | Path | None = None) -> dict[str, int]:
    manifest_path = Path(manifest_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    root = manifest_path.parent
    if rows:
        probe = root / rows[0]["target_path"]
        if not probe.exists():
            root = manifest_path.parent.parent
    color_set: set[RGB] = set()

    for row in rows:
        target_path = (root / row["target_path"]).resolve()
        image = _load_bitmap(target_path)
        width, height = image.size
        if width != 20 or height != 20:
            raise ValueError(f"Expected 20x20 instruction image, got {width}x{height} for {target_path}")
        for y_pos in range(height):
            for x_pos in range(width):
                color = tuple(int(channel) for channel in image.getpixel((x_pos, y_pos)))
                color_set.add(color)

    expected = set(OFFICIAL_PALETTE)
    if color_set != expected:
        missing = sorted(expected - color_set)
        extra = sorted(color_set - expected)
        raise ValueError(
            "Instruction palette does not match the official inverse knitting palette. "
            f"Missing colors: {missing}; extra colors: {extra}"
        )

    jsonable = official_palette_mapping()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(jsonable, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate dataset2 instruction palette and export the official inverse-knitting 17-color mapping.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest with target_path and index_path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON mapping")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mapping = infer_palette_mapping(args.manifest, args.output)
    print(json.dumps(mapping, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
