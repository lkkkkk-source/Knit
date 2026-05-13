from __future__ import annotations

import json
import math
from pathlib import Path


IGNORE_INDEX = -100
EXPECTED_DESCRIPTOR_DIM = 329
VALID_CANONICAL_MODES = ("full_masked", "bbox_crop")
REQUIRED_FOREGROUND_CACHE_SCHEMA_VERSION = "foreground_v1_full_masked_labelbalanced_transition_kmeans_grammar_bank_v1"
FORBIDDEN_FOREGROUND_CACHE_KEYS = frozenset(
    {
        "clustering_feature",
        "label_spatial_feature",
        "label_spatial_area_norm",
        "label_spatial_channel_balanced",
        "label_transition_feature",
    }
)
REQUIRED_FOREGROUND_CACHE_KEYS = (
    "descriptors_by_category",
    "descriptor_mean_by_category",
    "descriptor_std_by_category",
    "descriptor_global_mean",
    "descriptor_global_std",
    "category_foreground_area_stats",
)
REQUIRED_CENTROID_ENTRY_KEYS = (
    "centroid_fg_mask_prob",
    "centroid_fg_mask",
    "centroid_label_prob_16",
)


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_foreground_v1. Install with `pip install -e .[train]`.") from error


def load_config(path: str | Path) -> dict[str, object]:
    path = Path(path)
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required to load foreground_v1 config. Install with `pip install pyyaml`.") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return payload


def format_metric_line(prefix: str, items: list[tuple[str, object]]) -> str:
    parts = [prefix]
    for key, value in items:
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def resolve_canonical_mode(data_cf: dict[str, object]) -> str:
    if "canonical_mode" not in data_cf:
        raise ValueError("Config data.canonical_mode is required.")
    canonical_mode = str(data_cf["canonical_mode"])
    if canonical_mode not in VALID_CANONICAL_MODES:
        raise ValueError(
            f"Unsupported canonical_mode={canonical_mode!r}. "
            f"Expected one of {list(VALID_CANONICAL_MODES)}."
        )
    return canonical_mode


def require_ignore_index(data_cf: dict[str, object]) -> int:
    if "ignore_index" not in data_cf:
        raise ValueError("Config data.ignore_index is required.")
    ignore_index = int(data_cf["ignore_index"])
    if ignore_index != IGNORE_INDEX:
        raise ValueError(
            f"Config data.ignore_index must be {IGNORE_INDEX}, got {ignore_index}."
        )
    return ignore_index


def checkpoint_get(payload: dict[str, object], key: str, *, required: bool = True) -> object:
    if key in payload:
        return payload[key]
    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict) and key in metrics:
        return metrics[key]
    if required:
        raise ValueError(f"Checkpoint is missing required metadata field {key!r}.")
    return None


def _format_cache_field_path(parts: list[str | int]) -> str:
    text = ""
    for part in parts:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text = part if not text else f"{text}.{part}"
    return text or "<root>"


def assert_no_forbidden_cache_fields(
    payload: object,
    *,
    forbidden_keys: frozenset[str] = FORBIDDEN_FOREGROUND_CACHE_KEYS,
    context: str = "Foreground cache payload",
) -> None:
    def visit(value: object, path: list[str | int]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = [*path, str(key)]
                if isinstance(key, str) and key in forbidden_keys:
                    raise RuntimeError(
                        f"Forbidden large cache field found at {_format_cache_field_path(child_path)}"
                    )
                visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            if value and not isinstance(value[0], (dict, list, tuple)):
                return
            for index, child in enumerate(value):
                visit(child, [*path, index])

    try:
        visit(payload, [])
    except RuntimeError as error:
        raise RuntimeError(f"{context}: {error}") from error


def infer_model_kwargs_from_checkpoint_payload(payload: dict[str, object], config: dict[str, object]) -> tuple[dict[str, int], dict[str, object]]:
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")
    checkpoint_model_kwargs = payload.get("model_kwargs")
    source = "checkpoint_model_kwargs"
    if isinstance(checkpoint_model_kwargs, dict):
        model_kwargs = {key: int(value) for key, value in checkpoint_model_kwargs.items()}
    else:
        source = "state_dict_inference"

        def _shape(key: str) -> tuple[int, ...]:
            if key not in state_dict or not hasattr(state_dict[key], "shape"):
                raise ValueError(f"Checkpoint model_state_dict is missing tensor {key!r}.")
            return tuple(int(dim) for dim in state_dict[key].shape)

        category_embed_shape = _shape("category_embed.weight")
        mode_embed_shape = _shape("mode_embed.weight")
        grammar_head_shape = _shape("grammar_head.weight")
        adj_head_shape = _shape("adj_head.weight")
        bbox_head_shape = _shape("bbox_head.weight")
        if "condition_stem.0.weight" not in state_dict or "condition_stem.2.weight" not in state_dict:
            raise ValueError(
                "Checkpoint does not contain condition_stem.* weights, so it is not a compatible spatial-centroid foreground planner checkpoint. "
                "This loader only supports the current struct_foreground_v1 spatial-centroid architecture with strict=True loading."
            )
        condition_stem_shape = _shape("condition_stem.0.weight")
        condition_stem_hidden_shape = _shape("condition_stem.2.weight")
        fg_label_head_shape = _shape("fg_label_head.weight")
        fg_mask_head_shape = _shape("fg_mask_head.weight")
        if len(fg_label_head_shape) != 4 or fg_label_head_shape[0] != 16:
            raise ValueError(
                f"Checkpoint fg_label_head.weight must be conv-style with 16 output channels for current struct_foreground_v1 spatial-centroid planner, got {fg_label_head_shape}."
            )
        if len(fg_mask_head_shape) != 4 or fg_mask_head_shape[0] != 1:
            raise ValueError(
                f"Checkpoint fg_mask_head.weight must be conv-style with 1 output channel for current struct_foreground_v1 spatial-centroid planner, got {fg_mask_head_shape}."
            )
        model_kwargs = {
            "num_categories": category_embed_shape[0],
            "max_num_modes": mode_embed_shape[0],
            "hidden_dim": grammar_head_shape[1],
            "category_embed_dim": category_embed_shape[1],
            "mode_embed_dim": mode_embed_shape[1],
            "grammar_dim": grammar_head_shape[0],
            "adjacency_dim": adj_head_shape[0],
            "bbox_dim": bbox_head_shape[0],
            "spatial_condition_channels": condition_stem_shape[1],
            "spatial_hidden_dim": condition_stem_hidden_shape[0],
        }
    debug_info = {
        "source": source,
        "checkpoint_grammar_head_weight_shape": tuple(int(dim) for dim in state_dict["grammar_head.weight"].shape),
        "inferred_grammar_dim": int(model_kwargs["grammar_dim"]),
        "inferred_hidden_dim": int(model_kwargs["hidden_dim"]),
    }
    return model_kwargs, debug_info


def build_planner_from_checkpoint_payload(
    payload: dict[str, object],
    config: dict[str, object],
    *,
    device: object | None = None,
) -> tuple[object, dict[str, int], dict[str, object]]:
    from .models.foreground_planner import ForegroundCanonicalPlanner

    model_kwargs, debug_info = infer_model_kwargs_from_checkpoint_payload(payload, config)
    model = ForegroundCanonicalPlanner(**model_kwargs)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")
    checkpoint_shape = tuple(int(dim) for dim in state_dict["grammar_head.weight"].shape)
    constructed_shape = tuple(int(dim) for dim in model.grammar_head.weight.shape)
    if constructed_shape != checkpoint_shape:
        raise ValueError(
            f"Checkpoint/model grammar_head.weight mismatch before load: checkpoint={checkpoint_shape} constructed={constructed_shape}."
        )
    model.load_state_dict(state_dict, strict=True)
    if device is not None:
        model.to(device)
    debug_info["constructed_grammar_head_weight_shape"] = constructed_shape
    debug_info["strict_load_success"] = True
    return model, model_kwargs, debug_info


def require_foreground_cache_fields(
    cache_payload: dict[str, object],
    *,
    required_keys: tuple[str, ...] = REQUIRED_FOREGROUND_CACHE_KEYS,
    context: str = "Foreground cache",
) -> None:
    meta = cache_payload.get("meta")
    schema_version = meta.get("schema_version") if isinstance(meta, dict) else None
    if schema_version != REQUIRED_FOREGROUND_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"{context} has incompatible schema_version={schema_version!r}; "
            f"expected {REQUIRED_FOREGROUND_CACHE_SCHEMA_VERSION!r}. "
            "Please rebuild the foreground cache with the current build_foreground_cache.py."
        )
    assert_no_forbidden_cache_fields(cache_payload, context=context)
    missing = [key for key in required_keys if key not in cache_payload]
    if missing:
        raise ValueError(
            f"{context} is missing required fields: {', '.join(missing)}. "
            "Please rebuild the train foreground cache with the current build_foreground_cache.py."
        )


def require_centroid_sketch_fields(
    cache_payload: dict[str, object],
    *,
    context: str = "Foreground cache",
) -> None:
    centroid_by_category = cache_payload.get("centroid_sketch_by_category")
    if not isinstance(centroid_by_category, dict):
        raise ValueError(
            f"{context} is missing centroid_sketch_by_category. "
            "Please rebuild the train foreground cache with the current build_foreground_cache.py."
        )
    for category, centroid_entries in centroid_by_category.items():
        if not isinstance(centroid_entries, dict):
            raise ValueError(f"{context} centroid_sketch_by_category[{category!r}] must be a dict.")
        for local_z, centroid in centroid_entries.items():
            if not isinstance(centroid, dict):
                raise ValueError(f"{context} centroid_sketch_by_category[{category!r}][{local_z!r}] must be a dict.")
            missing = [key for key in REQUIRED_CENTROID_ENTRY_KEYS if key not in centroid]
            if missing:
                raise ValueError(
                    f"{context} centroid_sketch_by_category[{category!r}][{local_z!r}] is missing fields: {', '.join(missing)}. "
                    "Please rebuild the train foreground cache with the current build_foreground_cache.py."
                )
            centroid_fg_mask_prob = centroid["centroid_fg_mask_prob"]
            centroid_fg_mask = centroid["centroid_fg_mask"]
            centroid_label_prob_16 = centroid["centroid_label_prob_16"]
            if hasattr(centroid_fg_mask_prob, "tolist"):
                centroid_fg_mask_prob = centroid_fg_mask_prob.tolist()
            if hasattr(centroid_fg_mask, "tolist"):
                centroid_fg_mask = centroid_fg_mask.tolist()
            if hasattr(centroid_label_prob_16, "tolist"):
                centroid_label_prob_16 = centroid_label_prob_16.tolist()
            if not isinstance(centroid_fg_mask_prob, list) or len(centroid_fg_mask_prob) != 1:
                raise ValueError(f"{context} centroid_fg_mask_prob for category={category!r} local_z={local_z!r} must have shape [1,20,20].")
            if not isinstance(centroid_fg_mask_prob[0], list) or len(centroid_fg_mask_prob[0]) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in centroid_fg_mask_prob[0]):
                raise ValueError(f"{context} centroid_fg_mask_prob for category={category!r} local_z={local_z!r} must have shape [1,20,20].")
            if not isinstance(centroid_fg_mask, list) or len(centroid_fg_mask) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in centroid_fg_mask):
                raise ValueError(f"{context} centroid_fg_mask for category={category!r} local_z={local_z!r} must have shape [20,20].")
            if not isinstance(centroid_label_prob_16, list) or len(centroid_label_prob_16) != 16:
                raise ValueError(f"{context} centroid_label_prob_16 for category={category!r} local_z={local_z!r} must have shape [16,20,20].")
            for channel_index, channel in enumerate(centroid_label_prob_16):
                if not isinstance(channel, list) or len(channel) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in channel):
                    raise ValueError(
                        f"{context} centroid_label_prob_16 for category={category!r} local_z={local_z!r} "
                        f"channel={channel_index} must have shape [20,20]."
                    )


def print_progress(stage: str, current: int, total: int, extra: str = "") -> None:
    width = 30
    ratio = 0.0 if total <= 0 else current / total
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {extra}" if extra else ""
    print(f"\r[{stage}] [{bar}] {current}/{total}{suffix}", end="", flush=True)


def finish_progress() -> None:
    print(flush=True)


def save_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_jsonl(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _label_image_from_grid(mask: list[list[int]]) -> object:
    from knit_decode.parser_t_inverse.palette import OFFICIAL_PALETTE

    from PIL import Image
    height = len(mask)
    width = len(mask[0]) if height else 0
    image = Image.new("P", (width, height))
    for y_pos, row in enumerate(mask):
        for x_pos, class_id in enumerate(row):
            image.putpixel((x_pos, y_pos), int(class_id))
    palette: list[int] = []
    for color in OFFICIAL_PALETTE:
        palette.extend(color)
    palette.extend([0] * (768 - len(palette)))
    image.putpalette(palette)
    return image


def save_label_map(mask: list[list[int]], output_path: Path, scale: int = 1) -> None:
    try:
        image = _label_image_from_grid(mask)
    except Exception:
        save_json(output_path.with_suffix(".json"), mask)
        return
    if scale > 1:
        from PIL import Image

        width, height = image.size
        image = image.resize((width * scale, height * scale), resample=Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_label_grid_mosaic(grids: list[list[list[int]]], output_path: Path, *, columns: int = 4, scale: int = 1) -> None:
    if not grids:
        raise ValueError("save_label_grid_mosaic requires at least one grid.")
    try:
        from PIL import Image
    except Exception:
        save_json(output_path.with_suffix(".json"), {"columns": columns, "grids": grids})
        return
    images = [_label_image_from_grid(grid) for grid in grids]
    tile_width, tile_height = images[0].size
    columns = max(1, columns)
    rows = (len(images) + columns - 1) // columns
    mosaic = Image.new("P", (tile_width * columns, tile_height * rows))
    mosaic.putpalette(images[0].getpalette())
    for index, image in enumerate(images):
        x_offset = (index % columns) * tile_width
        y_offset = (index // columns) * tile_height
        mosaic.paste(image, (x_offset, y_offset))
    if scale > 1:
        mosaic = mosaic.resize((mosaic.size[0] * scale, mosaic.size[1] * scale), resample=Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(output_path)


def save_binary_map(mask: list[list[int]] | list[list[bool]], output_path: Path, scale: int = 1) -> None:
    try:
        from PIL import Image
    except Exception:
        save_json(output_path.with_suffix(".json"), [[1 if bool(value) else 0 for value in row] for row in mask])
        return
    height = len(mask)
    width = len(mask[0]) if height else 0
    image = Image.new("L", (width, height))
    for y_pos, row in enumerate(mask):
        for x_pos, value in enumerate(row):
            image.putpixel((x_pos, y_pos), 255 if bool(value) else 0)
    if scale > 1:
        image = image.resize((width * scale, height * scale), resample=Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def resolve_manifest_path(raw_path: str | Path, manifest_root: str | Path, *, sample_id: str, field_name: str) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = (Path(manifest_root) / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Missing {field_name} for sample_id={sample_id}: raw_path={raw_path!r} resolved_path={str(resolved)!r}"
        )
    return resolved


def load_label_grid(path: str | Path, *, sample_id: str | None = None) -> list[list[int]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        torch = _require_torch()
        payload = getattr(torch, "load")(path, map_location="cpu")
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        return [[int(value) for value in row] for row in payload]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [[int(value) for value in row] for row in payload]
    try:
        from PIL import Image
    except Exception as error:
        raise ImportError(f"Pillow is required to read image label grids from {path}") from error
    from knit_decode.parser_t_inverse.dataset import read_palette_mapping
    from knit_decode.parser_t_inverse.palette import official_palette_mapping

    try:
        with Image.open(path) as image:
            image.load()
            image = image.convert("RGB")
    except Exception as error:
        sample_text = f" sample_id={sample_id}" if sample_id is not None else ""
        raise ValueError(f"Failed to open label grid at {path}.{sample_text}") from error
    palette_path = path.parent.parent / "palette_mapping.json"
    if not palette_path.exists():
        palette_path = Path("dataset2/palette_mapping.json")
    palette_source = "official_palette_mapping() fallback"
    if palette_path.exists():
        try:
            mapping = read_palette_mapping(palette_path)
            palette_source = str(palette_path)
        except Exception as error:
            sample_text = f" sample_id={sample_id}" if sample_id is not None else ""
            raise ValueError(f"Failed to read palette mapping at {palette_path} for {path}.{sample_text}") from error
    else:
        mapping_payload = official_palette_mapping()
        mapping = {
            tuple(int(part) for part in key.split(",")): int(value) - 1
            for key, value in mapping_payload.items()
        }
    width, height = image.size
    grid: list[list[int]] = []
    for y_pos in range(height):
        row: list[int] = []
        for x_pos in range(width):
            color = tuple(int(channel) for channel in image.getpixel((x_pos, y_pos)))
            if color not in mapping:
                sample_text = f" sample_id={sample_id}" if sample_id is not None else ""
                raise ValueError(
                    f"Unknown palette color {color} at ({x_pos}, {y_pos}) in {path}; "
                    f"palette_source={palette_source}.{sample_text}"
                )
            row.append(int(mapping[color]))
        grid.append(row)
    return grid


def bbox_from_mask(mask: list[list[bool]]) -> dict[str, float]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    ys = [y_pos for y_pos in range(height) for x_pos in range(width) if mask[y_pos][x_pos]]
    xs = [x_pos for y_pos in range(height) for x_pos in range(width) if mask[y_pos][x_pos]]
    if not ys or not xs:
        return {
            "x0": 0.0,
            "y0": 0.0,
            "x1": 0.0,
            "y1": 0.0,
            "w": 0.0,
            "h": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "area_ratio": 0.0,
            "aspect_ratio": 0.0,
        }
    x0 = min(xs)
    x1 = max(xs)
    y0 = min(ys)
    y1 = max(ys)
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    total = max(1, width * height)
    return {
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "w": float(w),
        "h": float(h),
        "center_x": (x0 + x1 + 1) / 2.0,
        "center_y": (y0 + y1 + 1) / 2.0,
        "area_ratio": sum(1 for row in mask for value in row if value) / float(total),
        "aspect_ratio": w / float(max(1, h)),
    }


def canonicalize_foreground(
    grid20: list[list[int]],
    background_class_id: int,
    canonical_size: int = 20,
    *,
    canonical_mode: str = "full_masked",
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, object]:
    if canonical_mode not in VALID_CANONICAL_MODES:
        raise ValueError(
            f"Unsupported canonical_mode={canonical_mode!r}. "
            f"Expected one of {list(VALID_CANONICAL_MODES)}."
        )
    if ignore_index != IGNORE_INDEX:
        raise ValueError(f"canonicalize_foreground expects ignore_index={IGNORE_INDEX}, got {ignore_index}.")
    if canonical_mode == "full_masked":
        if len(grid20) != canonical_size or any(len(row) != canonical_size for row in grid20):
            raise ValueError(
                f"full_masked canonicalization expects {canonical_size}x{canonical_size} grid, "
                f"got {len(grid20)}x{len(grid20[0]) if grid20 else 0}."
            )
        fg_mask20 = [[1 if int(value) != background_class_id else 0 for value in row] for row in grid20]
        bbox = bbox_from_mask([[bool(value) for value in row] for row in fg_mask20])
        if bbox["area_ratio"] <= 0.0:
            fg_y20 = [[ignore_index for _ in range(canonical_size)] for _ in range(canonical_size)]
            return {
                "fg_y20": fg_y20,
                "fg_mask20": fg_mask20,
                "bbox": bbox,
                "is_empty_foreground": True,
                "crop_y": [],
                "crop_mask": [],
                "canonical_mode": canonical_mode,
            }
        fg_y20 = []
        for y_pos in range(canonical_size):
            row: list[int] = []
            for x_pos in range(canonical_size):
                value = int(grid20[y_pos][x_pos])
                row.append(value if value != background_class_id else ignore_index)
            fg_y20.append(row)
        validate_foreground_labels(fg_y20, fg_mask20, canonical_size=canonical_size, context="canonicalize_foreground(full_masked)")
        return {
            "fg_y20": fg_y20,
            "fg_mask20": fg_mask20,
            "bbox": bbox,
            "is_empty_foreground": False,
            "crop_y": [],
            "crop_mask": [],
            "canonical_mode": canonical_mode,
        }
    torch = _require_torch()
    functional = __import__("importlib").import_module("torch.nn.functional")
    fg_mask = [[value != background_class_id for value in row] for row in grid20]
    bbox = bbox_from_mask(fg_mask)
    if bbox["area_ratio"] <= 0.0:
        fg_y20 = [[ignore_index for _ in range(canonical_size)] for _ in range(canonical_size)]
        fg_mask20 = [[0 for _ in range(canonical_size)] for _ in range(canonical_size)]
        return {
            "fg_y20": fg_y20,
            "fg_mask20": fg_mask20,
            "bbox": bbox,
            "is_empty_foreground": True,
            "crop_y": [],
            "crop_mask": [],
            "canonical_mode": canonical_mode,
        }
    x0, y0, x1, y1 = int(bbox["x0"]), int(bbox["y0"]), int(bbox["x1"]), int(bbox["y1"])
    crop_y = [row[x0 : x1 + 1] for row in grid20[y0 : y1 + 1]]
    crop_mask = [row[x0 : x1 + 1] for row in fg_mask[y0 : y1 + 1]]
    crop_h = len(crop_y)
    crop_w = len(crop_y[0]) if crop_h else 0
    scale = min(canonical_size / float(max(1, crop_h)), canonical_size / float(max(1, crop_w)))
    new_h = max(1, min(canonical_size, int(round(crop_h * scale))))
    new_w = max(1, min(canonical_size, int(round(crop_w * scale))))
    label_tensor = getattr(torch, "tensor")(crop_y, dtype=getattr(torch, "float32")).unsqueeze(0).unsqueeze(0)
    mask_tensor = getattr(torch, "tensor")([[1.0 if value else 0.0 for value in row] for row in crop_mask], dtype=getattr(torch, "float32")).unsqueeze(0).unsqueeze(0)
    resized_labels = functional.interpolate(label_tensor, size=(new_h, new_w), mode="nearest").squeeze(0).squeeze(0).round().to(dtype=getattr(torch, "long"))
    resized_mask = functional.interpolate(mask_tensor, size=(new_h, new_w), mode="nearest").squeeze(0).squeeze(0) >= 0.5
    fg_y20 = [[ignore_index for _ in range(canonical_size)] for _ in range(canonical_size)]
    fg_mask20 = [[0 for _ in range(canonical_size)] for _ in range(canonical_size)]
    offset_y = (canonical_size - new_h) // 2
    offset_x = (canonical_size - new_w) // 2
    for y_pos in range(new_h):
        for x_pos in range(new_w):
            out_y = offset_y + y_pos
            out_x = offset_x + x_pos
            if bool(resized_mask[y_pos, x_pos].item()):
                fg_mask20[out_y][out_x] = 1
                fg_y20[out_y][out_x] = int(resized_labels[y_pos, x_pos].item())
    validate_foreground_labels(fg_y20, fg_mask20, canonical_size=canonical_size, context="canonicalize_foreground(bbox_crop)")
    return {
        "fg_y20": fg_y20,
        "fg_mask20": fg_mask20,
        "bbox": bbox,
        "is_empty_foreground": False,
        "crop_y": crop_y,
        "crop_mask": [[1 if value else 0 for value in row] for row in crop_mask],
        "canonical_mode": canonical_mode,
    }


def _row_projection(mask: list[list[int]]) -> list[float]:
    return [sum(int(value) for value in row) / float(max(1, len(row))) for row in mask]


def _col_projection(mask: list[list[int]]) -> list[float]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    return [sum(int(mask[y_pos][x_pos]) for y_pos in range(height)) / float(max(1, height)) for x_pos in range(width)]


def _run_stats(lines: list[list[int]]) -> list[float]:
    run_counts: list[float] = []
    mean_lengths: list[float] = []
    max_lengths: list[float] = []
    for line in lines:
        runs: list[int] = []
        current = 0
        for value in line:
            if int(value):
                current += 1
            elif current > 0:
                runs.append(current)
                current = 0
        if current > 0:
            runs.append(current)
        run_counts.append(float(len(runs)))
        mean_lengths.append(sum(runs) / float(max(1, len(runs))) if runs else 0.0)
        max_lengths.append(float(max(runs)) if runs else 0.0)
    return [
        sum(run_counts) / float(max(1, len(run_counts))),
        sum(mean_lengths) / float(max(1, len(mean_lengths))),
        sum(max_lengths) / float(max(1, len(max_lengths))),
    ]


def _adjacency_signature(labels: list[list[int]], mask: list[list[int]], num_labels: int = 16) -> list[float]:
    counts = [[0.0 for _ in range(num_labels)] for _ in range(num_labels)]
    height = len(labels)
    width = len(labels[0]) if height else 0
    total = 0.0
    for y_pos in range(height):
        for x_pos in range(width):
            if not mask[y_pos][x_pos]:
                continue
            src = int(labels[y_pos][x_pos]) - 1
            for next_y, next_x in ((y_pos + 1, x_pos), (y_pos, x_pos + 1)):
                if 0 <= next_y < height and 0 <= next_x < width and mask[next_y][next_x]:
                    dst = int(labels[next_y][next_x]) - 1
                    if 0 <= src < num_labels and 0 <= dst < num_labels:
                        counts[src][dst] += 1.0
                        counts[dst][src] += 1.0
                        total += 2.0
    if total <= 0:
        return [0.0 for _ in range(num_labels * num_labels)]
    return [value / total for row in counts for value in row]


def _label_hist(labels: list[list[int]], mask: list[list[int]], num_labels: int = 16) -> list[float]:
    counts = [0.0 for _ in range(num_labels)]
    total = 0.0
    for y_pos in range(len(labels)):
        for x_pos in range(len(labels[0])):
            if mask[y_pos][x_pos]:
                label = int(labels[y_pos][x_pos]) - 1
                if 0 <= label < num_labels:
                    counts[label] += 1.0
                    total += 1.0
    if total <= 0:
        return counts
    return [value / total for value in counts]


def _transition_2x2_stats(labels: list[list[int]], mask: list[list[int]]) -> list[float]:
    height = len(labels)
    width = len(labels[0]) if height else 0
    total = 0
    all_same = 0
    fg_mixed = 0
    unique_mean = 0.0
    diag_change = 0
    vertical_change = 0
    horizontal_change = 0
    for y_pos in range(max(0, height - 1)):
        for x_pos in range(max(0, width - 1)):
            block_mask = [mask[y_pos][x_pos], mask[y_pos][x_pos + 1], mask[y_pos + 1][x_pos], mask[y_pos + 1][x_pos + 1]]
            if not any(block_mask):
                continue
            total += 1
            block = [
                int(labels[y_pos][x_pos]) if block_mask[0] else 0,
                int(labels[y_pos][x_pos + 1]) if block_mask[1] else 0,
                int(labels[y_pos + 1][x_pos]) if block_mask[2] else 0,
                int(labels[y_pos + 1][x_pos + 1]) if block_mask[3] else 0,
            ]
            unique = {value for value in block if value > 0}
            unique_mean += len(unique)
            if len(unique) == 1 and unique:
                all_same += 1
            if len(unique) >= 2:
                fg_mixed += 1
            if block[0] != block[3] or block[1] != block[2]:
                diag_change += 1
            if block[0] != block[2] or block[1] != block[3]:
                vertical_change += 1
            if block[0] != block[1] or block[2] != block[3]:
                horizontal_change += 1
    denom = float(max(1, total))
    return [
        all_same / denom,
        fg_mixed / denom,
        unique_mean / denom,
        diag_change / denom,
        vertical_change / denom,
        horizontal_change / denom,
    ]


def foreground_descriptor(fg_y20: list[list[int]], fg_mask20: list[list[int]], bbox: dict[str, float]) -> dict[str, object]:
    row_projection = _row_projection(fg_mask20)
    col_projection = _col_projection(fg_mask20)
    row_runs = _run_stats(fg_mask20)
    col_runs = _run_stats([[fg_mask20[y_pos][x_pos] for y_pos in range(len(fg_mask20))] for x_pos in range(len(fg_mask20[0]))])
    adjacency_signature = _adjacency_signature(fg_y20, fg_mask20, num_labels=16)
    label_hist_16 = _label_hist(fg_y20, fg_mask20, num_labels=16)
    transition = _transition_2x2_stats(fg_y20, fg_mask20)
    height = len(fg_mask20)
    width = len(fg_mask20[0]) if height else 0
    vertical_pairs = 0
    vertical_same = 0
    horizontal_pairs = 0
    horizontal_same = 0
    for y_pos in range(height):
        for x_pos in range(width):
            if y_pos + 1 < height:
                vertical_pairs += 1
                if fg_mask20[y_pos][x_pos] and fg_mask20[y_pos + 1][x_pos]:
                    vertical_same += 1
            if x_pos + 1 < width:
                horizontal_pairs += 1
                if fg_mask20[y_pos][x_pos] and fg_mask20[y_pos][x_pos + 1]:
                    horizontal_same += 1
    vertical_continuity = vertical_same / float(max(1, vertical_pairs))
    horizontal_continuity = horizontal_same / float(max(1, horizontal_pairs))
    half = width // 2
    symmetry_hits = 0
    total_sym = 0
    for y_pos in range(height):
        for x_pos in range(half):
            total_sym += 1
            if fg_mask20[y_pos][x_pos] == fg_mask20[y_pos][width - 1 - x_pos]:
                symmetry_hits += 1
    symmetry_score = symmetry_hits / float(max(1, total_sym))
    center_start = int(width * 0.3)
    center_end = int(width * 0.7)
    center_fg = sum(int(fg_mask20[y_pos][x_pos]) for y_pos in range(height) for x_pos in range(center_start, center_end))
    total_fg = sum(int(value) for row in fg_mask20 for value in row)
    center_band_score = center_fg / float(max(1, total_fg))
    mean_col = sum(col_projection) / float(max(1, len(col_projection)))
    col_var = sum((value - mean_col) ** 2 for value in col_projection) / float(max(1, len(col_projection)))
    stripe_peaks = 0
    for index, value in enumerate(col_projection):
        left = col_projection[index - 1] if index > 0 else value
        right = col_projection[index + 1] if index + 1 < len(col_projection) else value
        if value > mean_col and value >= left and value >= right:
            stripe_peaks += 1
    stripe_score = [col_var, stripe_peaks / float(max(1, len(col_projection))), max(col_projection) if col_projection else 0.0]
    grammar_signature = row_runs + col_runs + transition + [vertical_continuity, horizontal_continuity, symmetry_score, center_band_score] + stripe_score
    descriptor = row_projection + col_projection + label_hist_16 + adjacency_signature + transition + [vertical_continuity, horizontal_continuity, symmetry_score, center_band_score] + stripe_score + [bbox["area_ratio"], bbox["aspect_ratio"], bbox["center_x"], bbox["center_y"]]
    descriptor_slices = {
        "row_projection": [0, 20],
        "col_projection": [20, 40],
        "label_hist_16": [40, 56],
        "adjacency_signature": [56, 312],
        "transition_2x2_stats": [312, 318],
        "grammar_signature_tail": [318, len(descriptor)],
    }
    return {
        "row_projection": row_projection,
        "col_projection": col_projection,
        "label_hist_16": label_hist_16,
        "adjacency_signature": adjacency_signature,
        "transition_2x2_stats": transition,
        "vertical_continuity": vertical_continuity,
        "horizontal_continuity": horizontal_continuity,
        "symmetry_score": symmetry_score,
        "center_band_score": center_band_score,
        "stripe_score": stripe_score,
        "grammar_signature": grammar_signature,
        "descriptor": descriptor,
        "descriptor_slices": descriptor_slices,
    }


def bbox_vector(bbox: dict[str, float], canonical_size: int = 20) -> list[float]:
    return [
        bbox["x0"] / float(canonical_size),
        bbox["y0"] / float(canonical_size),
        bbox["x1"] / float(canonical_size),
        bbox["y1"] / float(canonical_size),
        bbox["w"] / float(canonical_size),
        bbox["h"] / float(canonical_size),
        bbox["center_x"] / float(canonical_size),
        bbox["center_y"] / float(canonical_size),
        bbox["area_ratio"],
        bbox["aspect_ratio"],
    ]


def fg_mask_iou(pred_mask: list[list[int]], target_mask: list[list[int]]) -> float:
    inter = 0
    union = 0
    for y_pos in range(len(pred_mask)):
        for x_pos in range(len(pred_mask[0])):
            pred = int(pred_mask[y_pos][x_pos]) > 0
            tgt = int(target_mask[y_pos][x_pos]) > 0
            if pred and tgt:
                inter += 1
            if pred or tgt:
                union += 1
    return inter / float(max(1, union))


def foreground_area(mask: list[list[int]] | list[list[bool]]) -> float:
    height = len(mask)
    width = len(mask[0]) if height else 0
    total = max(1, height * width)
    return sum(1 for row in mask for value in row if bool(value)) / float(total)


def mask_component_stats(mask: list[list[int]] | list[list[bool]]) -> dict[str, float]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    visited = [[False for _ in range(width)] for _ in range(height)]
    component_sizes: list[int] = []
    for y_pos in range(height):
        for x_pos in range(width):
            if visited[y_pos][x_pos] or not bool(mask[y_pos][x_pos]):
                continue
            stack = [(y_pos, x_pos)]
            visited[y_pos][x_pos] = True
            size = 0
            while stack:
                cur_y, cur_x = stack.pop()
                size += 1
                for next_y, next_x in ((cur_y - 1, cur_x), (cur_y + 1, cur_x), (cur_y, cur_x - 1), (cur_y, cur_x + 1)):
                    if 0 <= next_y < height and 0 <= next_x < width and not visited[next_y][next_x] and bool(mask[next_y][next_x]):
                        visited[next_y][next_x] = True
                        stack.append((next_y, next_x))
            component_sizes.append(size)
    if not component_sizes:
        return {
            "num_components": 0.0,
            "largest_component_ratio": 0.0,
            "tiny_component_count": 0.0,
        }
    total_fg = sum(component_sizes)
    largest = max(component_sizes)
    tiny_count = sum(1 for size in component_sizes if size <= 4)
    return {
        "num_components": float(len(component_sizes)),
        "largest_component_ratio": float(largest) / float(max(1, total_fg)),
        "tiny_component_count": float(tiny_count),
    }


def label_spatial_feature_flat(
    fg_y20: list[list[int]],
    *,
    canonical_size: int = 20,
    num_labels: int = 16,
) -> list[float]:
    feature = [0.0 for _ in range(num_labels * canonical_size * canonical_size)]
    for y_pos in range(canonical_size):
        for x_pos in range(canonical_size):
            label_value = int(fg_y20[y_pos][x_pos])
            if 1 <= label_value <= num_labels:
                feature[((label_value - 1) * canonical_size * canonical_size) + (y_pos * canonical_size) + x_pos] = 1.0
    return feature


def label_spatial_feature_channels(
    fg_y20: list[list[int]],
    *,
    canonical_size: int = 20,
    num_labels: int = 16,
) -> list[list[list[float]]]:
    channels = [[[0.0 for _ in range(canonical_size)] for _ in range(canonical_size)] for _ in range(num_labels)]
    for y_pos in range(canonical_size):
        for x_pos in range(canonical_size):
            label_value = int(fg_y20[y_pos][x_pos])
            if 1 <= label_value <= num_labels:
                channels[label_value - 1][y_pos][x_pos] = 1.0
    return channels


def _flatten_channels(channels: list[list[list[float]]]) -> list[float]:
    return [float(value) for channel in channels for row in channel for value in row]


def _l2_normalize_feature_block(values: list[float], eps: float = 1e-12) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= eps:
        return [0.0 for _ in values]
    return [float(value) / norm for value in values]


def label_transition_edge_feature_flat(
    fg_y20: list[list[int]],
    *,
    canonical_size: int = 20,
    num_labels: int = 16,
) -> list[float]:
    channels = [[[0.0 for _ in range(canonical_size)] for _ in range(canonical_size)] for _ in range(num_labels)]
    for y_pos in range(canonical_size):
        for x_pos in range(canonical_size):
            label_value = int(fg_y20[y_pos][x_pos])
            if not (1 <= label_value <= num_labels):
                continue
            has_transition = False
            for next_y, next_x in ((y_pos - 1, x_pos), (y_pos + 1, x_pos), (y_pos, x_pos - 1), (y_pos, x_pos + 1)):
                if 0 <= next_y < canonical_size and 0 <= next_x < canonical_size:
                    next_label = int(fg_y20[next_y][next_x])
                    if 1 <= next_label <= num_labels and next_label != label_value:
                        has_transition = True
                        break
            if has_transition:
                channels[label_value - 1][y_pos][x_pos] = 1.0
    return _flatten_channels(channels)


def clustering_feature_from_parts(
    fg_y20: list[list[int]],
    fg_mask20: list[list[int]] | list[list[bool]],
    bbox_stats: list[float],
    row_projection: list[float],
    col_projection: list[float],
    *,
    canonical_size: int = 20,
    num_labels: int = 16,
    label_spatial_feature_weight: float = 0.5,
    label_spatial_area_norm_weight: float = 1.0,
    label_spatial_channel_balanced_weight: float = 1.0,
    label_transition_feature_weight: float = 0.75,
    mask_feature_weight: float = 0.05,
    row_col_feature_weight: float = 0.05,
    bbox_feature_weight: float = 0.0,
    eps: float = 1e-12,
) -> dict[str, object]:
    label_spatial_channels = label_spatial_feature_channels(fg_y20, canonical_size=canonical_size, num_labels=num_labels)
    label_spatial_feature = _flatten_channels(label_spatial_channels)
    fg_area = sum(float(value) for row in fg_mask20 for value in row)
    fg_area_denom = math.sqrt(max(fg_area, 1.0))
    label_spatial_area_norm = [float(value) / fg_area_denom for value in label_spatial_feature]
    label_spatial_channel_balanced_channels: list[list[list[float]]] = []
    for channel in label_spatial_channels:
        channel_mass = sum(float(value) for row in channel for value in row)
        denom = math.sqrt(channel_mass + eps)
        label_spatial_channel_balanced_channels.append([[float(value) / denom for value in row] for row in channel])
    label_spatial_channel_balanced = _flatten_channels(label_spatial_channel_balanced_channels)
    label_transition_feature = label_transition_edge_feature_flat(fg_y20, canonical_size=canonical_size, num_labels=num_labels)
    mask_feature = [float(int(value) > 0) for row in fg_mask20 for value in row]
    row_feature = [float(value) for value in row_projection]
    col_feature = [float(value) for value in col_projection]
    row_col_feature = row_feature + col_feature
    bbox_feature = [float(value) for value in bbox_stats]
    block_payloads = {
        "label_spatial_feature": (_l2_normalize_feature_block(label_spatial_feature, eps=eps), float(label_spatial_feature_weight)),
        "label_spatial_area_norm": (_l2_normalize_feature_block(label_spatial_area_norm, eps=eps), float(label_spatial_area_norm_weight)),
        "label_spatial_channel_balanced": (_l2_normalize_feature_block(label_spatial_channel_balanced, eps=eps), float(label_spatial_channel_balanced_weight)),
        "label_transition_feature": (_l2_normalize_feature_block(label_transition_feature, eps=eps), float(label_transition_feature_weight)),
        "mask_feature": (_l2_normalize_feature_block(mask_feature, eps=eps), float(mask_feature_weight)),
        "row_col_feature": (_l2_normalize_feature_block(row_col_feature, eps=eps), float(row_col_feature_weight)),
        "bbox_feature": (_l2_normalize_feature_block(bbox_feature, eps=eps), float(bbox_feature_weight)),
    }
    weighted_blocks: dict[str, list[float]] = {}
    for name, (block, weight) in block_payloads.items():
        weighted_blocks[name] = [float(value) * weight for value in block]
    clustering_feature = []
    clustering_feature_slices: dict[str, list[int]] = {}
    offset = 0
    for name in [
        "label_spatial_feature",
        "label_spatial_area_norm",
        "label_spatial_channel_balanced",
        "label_transition_feature",
        "mask_feature",
        "row_col_feature",
        "bbox_feature",
    ]:
        block = weighted_blocks[name]
        clustering_feature.extend(block)
        clustering_feature_slices[name] = [offset, offset + len(block)]
        offset += len(block)
    clustering_feature = _l2_normalize_feature_block(clustering_feature, eps=eps)
    final_norm = math.sqrt(sum(float(value) * float(value) for value in clustering_feature))
    return {
        "label_spatial_feature": label_spatial_feature,
        "label_spatial_area_norm": label_spatial_area_norm,
        "label_spatial_channel_balanced": label_spatial_channel_balanced,
        "label_transition_feature": label_transition_feature,
        "clustering_feature": clustering_feature,
        "clustering_feature_slices": clustering_feature_slices,
        "clustering_feature_block_lengths": {name: len(values) for name, values in {
            "label_spatial_feature": label_spatial_feature,
            "label_spatial_area_norm": label_spatial_area_norm,
            "label_spatial_channel_balanced": label_spatial_channel_balanced,
            "label_transition_feature": label_transition_feature,
            "mask_feature": mask_feature,
            "row_col_feature": row_col_feature,
            "bbox_feature": bbox_feature,
        }.items()},
        "clustering_feature_weights": {
            "label_spatial_feature_weight": float(label_spatial_feature_weight),
            "label_spatial_area_norm_weight": float(label_spatial_area_norm_weight),
            "label_spatial_channel_balanced_weight": float(label_spatial_channel_balanced_weight),
            "label_transition_feature_weight": float(label_transition_feature_weight),
            "mask_feature_weight": float(mask_feature_weight),
            "row_col_feature_weight": float(row_col_feature_weight),
            "bbox_feature_weight": float(bbox_feature_weight),
        },
        "clustering_feature_flags": {
            "area_norm": True,
            "channel_balance": True,
            "transition_feature": True,
            "final_l2_normalize": True,
        },
        "clustering_feature_final_norm": final_norm,
    }


def label_diversity_on_fg(labels: list[list[int]], mask: list[list[int]] | list[list[bool]]) -> int:
    values = {
        int(labels[y_pos][x_pos])
        for y_pos in range(len(labels))
        for x_pos in range(len(labels[0]))
        if bool(mask[y_pos][x_pos]) and 1 <= int(labels[y_pos][x_pos]) <= 16
    }
    return len(values)


def validate_foreground_labels(
    fg_y20: list[list[int]],
    fg_mask20: list[list[int]] | list[list[bool]],
    *,
    canonical_size: int = 20,
    context: str = "foreground",
) -> None:
    if len(fg_y20) != canonical_size or len(fg_mask20) != canonical_size:
        raise ValueError(f"{context}: expected {canonical_size} rows for fg_y20 and fg_mask20.")
    for row_index in range(canonical_size):
        if len(fg_y20[row_index]) != canonical_size or len(fg_mask20[row_index]) != canonical_size:
            raise ValueError(f"{context}: expected {canonical_size} columns for fg_y20 and fg_mask20.")
        for col_index in range(canonical_size):
            label_value = int(fg_y20[row_index][col_index])
            mask_value = bool(fg_mask20[row_index][col_index])
            if mask_value:
                if not (1 <= label_value <= 16):
                    raise ValueError(
                        f"{context}: foreground pixel at ({row_index}, {col_index}) has invalid label {label_value}; expected 1..16."
                    )
            else:
                if label_value != IGNORE_INDEX:
                    raise ValueError(
                        f"{context}: background pixel at ({row_index}, {col_index}) must be IGNORE_INDEX={IGNORE_INDEX}, got {label_value}."
                    )


def ensure_descriptor_dim(descriptor: list[float], *, context: str) -> None:
    if len(descriptor) != EXPECTED_DESCRIPTOR_DIM:
        raise ValueError(f"{context}: expected descriptor dim {EXPECTED_DESCRIPTOR_DIM}, got {len(descriptor)}.")


def descriptor_stats_by_category(items: list[dict[str, object]], categories: list[str]) -> tuple[dict[str, list[list[float]]], dict[str, list[float]], dict[str, list[float]], dict[str, dict[str, float]]]:
    descriptors_by_category: dict[str, list[list[float]]] = {}
    descriptor_mean_by_category: dict[str, list[float]] = {}
    descriptor_std_by_category: dict[str, list[float]] = {}
    category_foreground_area_stats: dict[str, dict[str, float]] = {}
    for category in categories:
        descs = [item["descriptor"] for item in items if item["category"] == category and not item["is_empty_foreground"]]
        for descriptor in descs:
            ensure_descriptor_dim(descriptor, context=f"descriptor_stats_by_category[{category}]")
        descriptors_by_category[category] = descs
        if not descs:
            descriptor_mean_by_category[category] = []
            descriptor_std_by_category[category] = []
            category_foreground_area_stats[category] = {
                "count": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "q01": 0.0,
                "q05": 0.0,
                "q10": 0.0,
                "q50": 0.0,
                "q90": 0.0,
                "q95": 0.0,
                "q99": 0.0,
                "valid_low": 0.02,
                "valid_high": 0.98,
            }
            continue
        dim = len(descs[0])
        means = []
        stds = []
        for dim_index in range(dim):
            values = [float(descriptor[dim_index]) for descriptor in descs]
            mean_value = sum(values) / float(len(values))
            std_value = (sum((value - mean_value) ** 2 for value in values) / float(max(1, len(values)))) ** 0.5
            means.append(mean_value)
            stds.append(max(std_value, 1e-6))
        descriptor_mean_by_category[category] = means
        descriptor_std_by_category[category] = stds
        areas = sorted(float(item["fg_area"]) for item in items if item["category"] == category and not item["is_empty_foreground"])

        def _q(q: float) -> float:
            index = min(len(areas) - 1, max(0, int(round((len(areas) - 1) * q))))
            return areas[index]

        mean_area = sum(areas) / float(len(areas))
        std_area = (sum((value - mean_area) ** 2 for value in areas) / float(max(1, len(areas)))) ** 0.5
        category_foreground_area_stats[category] = {
            "count": float(len(areas)),
            "mean": mean_area,
            "std": std_area,
            "q01": _q(0.01),
            "q05": _q(0.05),
            "q10": _q(0.10),
            "q50": _q(0.50),
            "q90": _q(0.90),
            "q95": _q(0.95),
            "q99": _q(0.99),
            "valid_low": max(0.02, _q(0.05) - 0.02),
            "valid_high": min(0.98, _q(0.95) + 0.02),
        }
    return descriptors_by_category, descriptor_mean_by_category, descriptor_std_by_category, category_foreground_area_stats


def descriptor_global_stats(items: list[dict[str, object]]) -> tuple[list[float], list[float]]:
    descs = [item["descriptor"] for item in items if not item.get("is_empty_foreground", False)]
    if not descs:
        return [], []
    for descriptor in descs:
        ensure_descriptor_dim(descriptor, context="descriptor_global_stats")
    dim = len(descs[0])
    means: list[float] = []
    stds: list[float] = []
    for dim_index in range(dim):
        values = [float(descriptor[dim_index]) for descriptor in descs]
        mean_value = sum(values) / float(len(values))
        std_value = (sum((value - mean_value) ** 2 for value in values) / float(max(1, len(values)))) ** 0.5
        means.append(mean_value)
        stds.append(max(std_value, 1e-6))
    return means, stds


def normalized_l2(descriptor: list[float], mean: list[float], std: list[float]) -> float:
    if not mean or not std:
        return float("inf")
    total = 0.0
    for value, mean_value, std_value in zip(descriptor, mean, std):
        total += ((float(value) - float(mean_value)) / max(float(std_value), 1e-6)) ** 2
    return math.sqrt(total / float(max(1, len(descriptor))))


def normalized_l2_between(descriptor: list[float], reference: list[float], global_mean: list[float], global_std: list[float]) -> float:
    if not reference or not global_mean or not global_std:
        return float("inf")
    total = 0.0
    dim = min(len(descriptor), len(reference), len(global_mean), len(global_std))
    for index in range(dim):
        desc_value = (float(descriptor[index]) - float(global_mean[index])) / max(float(global_std[index]), 1e-6)
        ref_value = (float(reference[index]) - float(global_mean[index])) / max(float(global_std[index]), 1e-6)
        total += (desc_value - ref_value) ** 2
    return math.sqrt(total / float(max(1, dim)))
