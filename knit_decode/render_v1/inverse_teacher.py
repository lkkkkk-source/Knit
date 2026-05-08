from __future__ import annotations

from pathlib import Path

from knit_decode.parser_t_inverse.losses import build_syntax_penalties, syntax_loss
from knit_decode.parser_t_inverse.model import InverseImg2Prog


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for inverse teacher loading. Install with `pip install -e .[train]`.") from error


class FrozenInverseTeacher:
    def __init__(self, checkpoint_path: str | Path, syntax_dir: str | Path, device: object) -> None:
        torch = _require_torch()
        checkpoint = getattr(torch, "load")(Path(checkpoint_path), map_location="cpu")
        metrics = checkpoint.get("metrics", {})
        self.model = InverseImg2Prog(num_classes=17)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.image_size = tuple(int(value) for value in metrics.get("image_size", [160, 160]))
        self.syntax_penalties = build_syntax_penalties(syntax_dir, num_classes=17)

    def logits(self, images: object) -> object:
        if images.shape[-2:] != self.image_size:
            functional = __import__("importlib").import_module("torch.nn.functional")
            images = functional.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
        grayscale = images.mean(dim=1, keepdim=True)
        grayscale = (grayscale + 1.0) * 0.5
        return self.model(grayscale)

    def syntax_loss(self, logits: object) -> object:
        return syntax_loss(logits, self.syntax_penalties)
