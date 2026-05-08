from __future__ import annotations

from pathlib import Path

from knit_decode.parser_t_inverse.model import InverseImg2Prog


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for refiner_inverse_v1 teacher loading. Install with `pip install -e .[train]`.") from error


def build_trainable_parser(checkpoint_path: str | Path | None, device: object) -> object:
    model = InverseImg2Prog(num_classes=17)
    model.to(device)
    if checkpoint_path is not None:
        torch = _require_torch()
        checkpoint = getattr(torch, "load")(Path(checkpoint_path), map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    return model
