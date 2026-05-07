from __future__ import annotations

import json
from pathlib import Path

from .dataset import _require_torch as _require_torch_render


def _require_torch() -> object:
    torch, _ = _require_torch_render()
    return torch


class FrozenParserTeacher:
    def __init__(self, checkpoint_path: str | Path, device: object) -> None:
        from knit_decode.parser_t.model import build_parser_model

        torch = _require_torch()
        checkpoint = getattr(torch, "load")(Path(checkpoint_path), map_location="cpu")
        metrics = checkpoint.get("metrics", {})
        num_classes = int(metrics.get("num_classes", 3))
        model_name = str(metrics.get("model", "kaspar"))
        self.model = build_parser_model(model_name, num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.num_classes = num_classes
        self.image_size = tuple(int(value) for value in metrics.get("image_size", [160, 160]))
        self.grid_size = tuple(int(value) for value in metrics.get("grid_size", [20, 20]))

    def __call__(self, images: object) -> object:
        torch = _require_torch()
        if images.shape[-2:] != self.image_size:
            interpolate = getattr(__import__("importlib").import_module("torch.nn.functional"), "interpolate")
            images = interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
        grayscale = images.mean(dim=1, keepdim=True)
        grayscale = (grayscale + 1.0) * 0.5
        return self.model(grayscale)
