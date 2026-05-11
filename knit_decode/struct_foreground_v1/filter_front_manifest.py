from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import format_metric_line, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter a manifest down to front-only instruction samples.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background-rgb", type=str, default="255,0,16")
    parser.add_argument("--drop-if-path-contains", type=str, default="back")
    parser.add_argument("--check-border-color", type=str, default="true")
    parser.add_argument("--border-color-tolerance", type=int, default=0)
    parser.add_argument("--max-report", type=int, default=50)
    parser.add_argument("--strict-missing-files", type=str, default="true")
    return parser


def _parse_bool(text: str) -> bool:
    lowered = str(text).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean string, got {text!r}")


def _parse_rgb(text: str) -> list[int]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB string like '255,0,16', got {text!r}")
    rgb = [int(part) for part in parts]
    for value in rgb:
        if value < 0 or value > 255:
            raise ValueError(f"RGB values must be in [0,255], got {rgb}")
    return rgb


def _load_manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_target_path(manifest_path: Path, raw_path: str, sample_id: str) -> Path:
    candidates = [
        (Path.cwd() / raw_path).resolve(),
        (manifest_path.parent / raw_path).resolve(),
        (manifest_path.parent.parent / raw_path).resolve() if manifest_path.parent != manifest_path.parent.parent else None,
        (manifest_path.parent.parent.parent / raw_path).resolve() if manifest_path.parent.parent != manifest_path.parent.parent.parent else None,
    ]
    tried = [str(candidate) for candidate in candidates if candidate is not None]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve target_path for sample_id={sample_id}: raw target_path={raw_path!r}; tried={tried}"
    )


def _read_rgb_image(path: Path) -> list[list[list[int]]]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            image = image.convert("RGB")
            width, height = image.size
            return [
                [[int(channel) for channel in image.getpixel((x_pos, y_pos))] for x_pos in range(width)]
                for y_pos in range(height)
            ]
    except Exception:
        try:
            import cv2

            payload = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if payload is None:
                raise ValueError(f"cv2.imread returned None for {path}")
            payload = cv2.cvtColor(payload, cv2.COLOR_BGR2RGB)
            return [
                [[int(payload[y_pos, x_pos, channel]) for channel in range(3)] for x_pos in range(payload.shape[1])]
                for y_pos in range(payload.shape[0])
            ]
        except Exception:
            pass
        import imageio.v2 as imageio

        payload = imageio.imread(path)
        if len(payload.shape) < 3 or payload.shape[2] < 3:
            raise ValueError(f"Expected RGB image at {path}, got shape {getattr(payload, 'shape', None)}")
        return [
            [[int(payload[y_pos, x_pos, channel]) for channel in range(3)] for x_pos in range(payload.shape[1])]
            for y_pos in range(payload.shape[0])
        ]


def _dominant_border_color(rgb_grid: list[list[list[int]]]) -> list[int]:
    height = len(rgb_grid)
    width = len(rgb_grid[0]) if height else 0
    border_pixels: list[tuple[int, int, int]] = []
    if height <= 0 or width <= 0:
        return [0, 0, 0]
    border_pixels.extend(tuple(pixel) for pixel in rgb_grid[0])
    border_pixels.extend(tuple(pixel) for pixel in rgb_grid[-1])
    border_pixels.extend(tuple(rgb_grid[y_pos][0]) for y_pos in range(height))
    border_pixels.extend(tuple(rgb_grid[y_pos][-1]) for y_pos in range(height))
    counts: dict[tuple[int, int, int], int] = {}
    for pixel in border_pixels:
        counts[pixel] = counts.get(pixel, 0) + 1
    dominant = max(counts.items(), key=lambda item: item[1])[0]
    return [int(part) for part in dominant]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    background_rgb = _parse_rgb(args.background_rgb)
    drop_token = str(args.drop_if_path_contains).lower()
    check_border_color = _parse_bool(args.check_border_color)
    strict_missing_files = _parse_bool(args.strict_missing_files)
    tolerance = int(args.border_color_tolerance)
    rows = _load_manifest(args.manifest)
    kept_rows: list[dict[str, object]] = []
    dropped_examples: list[dict[str, object]] = []
    histogram: dict[str, int] = {}
    dropped_by_contains_back = 0
    dropped_by_non_standard_border = 0
    dropped_by_both = 0

    for row in rows:
        sample_id = str(row["sample_id"])
        input_path = str(row.get("input_path", ""))
        target_path = str(row.get("target_path", ""))
        index_path = str(row.get("index_path", ""))
        joined = " ".join([sample_id, input_path, target_path, index_path]).lower()
        contains_back = drop_token in joined if drop_token else False
        resolved_target_path = None
        dominant_border_color = [0, 0, 0]
        non_standard_border = False
        try:
            resolved_target_path = _resolve_target_path(args.manifest, target_path, sample_id)
            if check_border_color:
                rgb_grid = _read_rgb_image(resolved_target_path)
                dominant_border_color = _dominant_border_color(rgb_grid)
                histogram_key = ",".join(str(part) for part in dominant_border_color)
                histogram[histogram_key] = histogram.get(histogram_key, 0) + 1
                non_standard_border = any(abs(int(a) - int(b)) > tolerance for a, b in zip(dominant_border_color, background_rgb))
        except FileNotFoundError:
            if strict_missing_files:
                raise
            non_standard_border = True
        reason = None
        if contains_back and non_standard_border:
            reason = "contains_back+non_standard_border"
            dropped_by_both += 1
        elif contains_back:
            reason = "contains_back"
            dropped_by_contains_back += 1
        elif non_standard_border:
            reason = "non_standard_border"
            dropped_by_non_standard_border += 1
        if reason is None:
            kept_rows.append(row)
            continue
        if len(dropped_examples) < int(args.max_report):
            dropped_examples.append(
                {
                    "sample_id": sample_id,
                    "input_path": input_path,
                    "target_path": target_path,
                    "index_path": index_path,
                    "reason": reason,
                    "contains_back": bool(contains_back),
                    "dominant_border_color": dominant_border_color,
                    "background_rgb": background_rgb,
                    "resolved_target_path": str(resolved_target_path) if resolved_target_path is not None else None,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept_rows), encoding="utf-8")
    report = {
        "input_manifest": str(args.manifest),
        "output_manifest": str(args.output),
        "total": len(rows),
        "kept": len(kept_rows),
        "dropped": len(rows) - len(kept_rows),
        "dropped_by_contains_back": dropped_by_contains_back,
        "dropped_by_non_standard_border": dropped_by_non_standard_border,
        "dropped_by_both": dropped_by_both,
        "background_rgb": background_rgb,
        "drop_if_path_contains": drop_token,
        "check_border_color": bool(check_border_color),
        "border_color_tolerance": tolerance,
        "dominant_border_color_histogram": histogram,
        "dropped_examples": dropped_examples,
    }
    report_path = Path(str(args.output) + ".filter_report.json")
    save_json(report_path, report)
    print(
        format_metric_line(
            "filter-front-manifest:",
            [
                ("input", str(args.manifest)),
                ("output", str(args.output)),
                ("total", len(rows)),
                ("kept", len(kept_rows)),
                ("dropped", len(rows) - len(kept_rows)),
                ("contains_back", dropped_by_contains_back),
                ("non_standard_border", dropped_by_non_standard_border),
                ("both", dropped_by_both),
                ("background_rgb", background_rgb),
                ("border_tolerance", tolerance),
            ],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
