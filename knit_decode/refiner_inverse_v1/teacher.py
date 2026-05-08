from __future__ import annotations

from pathlib import Path

from knit_decode.parser_t_inverse.model import InverseImg2Prog


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for refiner_inverse_v1 teacher loading. Install with `pip install -e .[train]`.") from error


class FrozenInverseParser:
    def __init__(self, checkpoint_path: str | Path, device: object) -> None:
        torch = _require_torch()
        checkpoint = getattr(torch, "load")(Path(checkpoint_path), map_location="cpu")
        self.model = InverseImg2Prog(num_classes=17)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(self, image: object) -> object:
        return self.model(image)
