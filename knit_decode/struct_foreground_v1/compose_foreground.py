from __future__ import annotations

from pathlib import Path

from .utils import resolve_canonical_mode, save_binary_map, save_label_map


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for compose_foreground.") from error


def compose_foreground(
    fg_mask20: list[list[int]],
    fg_label20: list[list[int]],
    bbox_pred: list[float],
    output_dir: str | Path | None = None,
    background_class_id: int = 0,
    canonical_mode: str = "full_masked",
) -> dict[str, object]:
    if len(fg_mask20) != 20 or len(fg_label20) != 20:
        raise ValueError("compose_foreground expects 20x20 fg_mask20 and fg_label20.")
    for row_index in range(20):
        if len(fg_mask20[row_index]) != 20 or len(fg_label20[row_index]) != 20:
            raise ValueError("compose_foreground expects 20x20 fg_mask20 and fg_label20.")
    canvas = [[background_class_id for _ in range(20)] for _ in range(20)]
    if canonical_mode == "full_masked":
        for y_pos in range(20):
            for x_pos in range(20):
                if int(fg_mask20[y_pos][x_pos]) > 0:
                    label_value = int(fg_label20[y_pos][x_pos])
                    if not (1 <= label_value <= 16):
                        raise ValueError(f"compose_foreground produced invalid foreground label {label_value}; expected 1..16.")
                    canvas[y_pos][x_pos] = label_value
        result = {
            "composed_y20": canvas,
            "bbox": {
                "x0": 0,
                "y0": 0,
                "x1": 19,
                "y1": 19,
                "w": 20,
                "h": 20,
            },
        }
    elif canonical_mode == "bbox_crop":
        torch = _require_torch()
        functional = __import__("importlib").import_module("torch.nn.functional")
        if len(bbox_pred) < 6:
            raise ValueError(f"compose_foreground requires bbox_pred with at least 6 values, got {len(bbox_pred)}.")
        x0 = max(0, min(19, int(round(bbox_pred[0] * 20.0))))
        y0 = max(0, min(19, int(round(bbox_pred[1] * 20.0))))
        w = max(1, min(20, int(round(bbox_pred[4] * 20.0))))
        h = max(1, min(20, int(round(bbox_pred[5] * 20.0))))
        x1 = min(19, x0 + w - 1)
        y1 = min(19, y0 + h - 1)
        mask_tensor = getattr(torch, "tensor")(fg_mask20, dtype=getattr(torch, "float32")).unsqueeze(0).unsqueeze(0)
        label_tensor = getattr(torch, "tensor")(fg_label20, dtype=getattr(torch, "float32")).unsqueeze(0).unsqueeze(0)
        resized_mask = functional.interpolate(mask_tensor, size=(y1 - y0 + 1, x1 - x0 + 1), mode="nearest").squeeze(0).squeeze(0)
        resized_label = functional.interpolate(label_tensor, size=(y1 - y0 + 1, x1 - x0 + 1), mode="nearest").squeeze(0).squeeze(0).round().to(dtype=getattr(torch, "long"))
        for y_pos in range(y1 - y0 + 1):
            for x_pos in range(x1 - x0 + 1):
                if float(resized_mask[y_pos, x_pos].item()) >= 0.5:
                    label_value = int(resized_label[y_pos, x_pos].item())
                    if not (1 <= label_value <= 16):
                        raise ValueError(f"compose_foreground produced invalid foreground label {label_value}; expected 1..16.")
                    canvas[y0 + y_pos][x0 + x_pos] = label_value
        result = {
            "composed_y20": canvas,
            "bbox": {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "w": x1 - x0 + 1,
                "h": y1 - y0 + 1,
            },
        }
    else:
        raise ValueError(f"Unsupported canonical_mode={canonical_mode!r} in compose_foreground.")
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_label_map(canvas, out / "composed_y20.png", scale=12)
        save_label_map(fg_label20, out / "fg_label20.png", scale=12)
        save_binary_map(fg_mask20, out / "fg_mask20.png", scale=12)
    return result
