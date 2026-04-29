from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def _load_action_colors(vocab_path: Path) -> dict[int, tuple[int, int, int]]:
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    actions = payload["actions"]
    color_map: dict[int, tuple[int, int, int]] = {}
    for entry in actions:
        action_id = int(entry["action_id"])
        color_hex = str(entry["color_hex"])
        color_map[action_id] = (
            int(color_hex[1:3], 16),
            int(color_hex[3:5], 16),
            int(color_hex[5:7], 16),
        )
    return color_map


def _render_sample(grid: list[list[int | None]], color_map: dict[int, tuple[int, int, int]], scale: int) -> Image.Image:
    rows = len(grid)
    columns = len(grid[0]) if rows else 0
    image = Image.new("RGB", (columns, rows), (0, 0, 0))
    for y_pos, row in enumerate(grid):
        for x_pos, token in enumerate(row):
            color = color_map.get(int(token), (255, 255, 255)) if token is not None else (255, 255, 255)
            image.putpixel((x_pos, y_pos), color)
    if scale > 1:
        image = image.resize((columns * scale, rows * scale), resample=Image.NEAREST)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Render generated raw action grids into color-code PNGs")
    parser.add_argument("--generated-json", type=Path, required=True, help="Path to generated_grids.json")
    parser.add_argument("--vocab-json", type=Path, required=True, help="Path to ar_vocab.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for rendered color maps")
    parser.add_argument("--scale", type=int, default=16, help="Nearest-neighbor scale factor for visualization")
    args = parser.parse_args()

    payload = json.loads(args.generated_json.read_text(encoding="utf-8"))
    raw_samples = payload["raw_action_samples"]
    color_map = _load_action_colors(args.vocab_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample_index, sample_grid in enumerate(raw_samples):
        image = _render_sample(sample_grid, color_map, scale=args.scale)
        image.save(args.output_dir / f"sample_{sample_index:03d}.png")

    summary = {
        "generated_json": str(args.generated_json),
        "vocab_json": str(args.vocab_json),
        "output_dir": str(args.output_dir),
        "samples": len(raw_samples),
        "scale": args.scale,
    }
    (args.output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
