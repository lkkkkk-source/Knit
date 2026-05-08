from __future__ import annotations

from typing import Sequence


def _require_torch_and_vision() -> tuple[object, object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        models = importlib.import_module("torchvision.models")
    except ImportError as error:
        raise ImportError(
            "torchvision is required for VGG perceptual/style loss in inverse_residual_v1. "
            "Install with `pip install torchvision`."
        ) from error
    return torch, nn, models


class VGGFeatureExtractor:
    """Small wrapper around torchvision VGG features for perceptual/style losses."""

    def __init__(self, variant: str = "16", device: object | None = None) -> None:
        torch, nn, models = _require_torch_and_vision()
        if variant == "16":
            weights = getattr(models, "VGG16_Weights").IMAGENET1K_V1
            features = models.vgg16(weights=weights).features
        elif variant == "19":
            weights = getattr(models, "VGG19_Weights").IMAGENET1K_V1
            features = models.vgg19(weights=weights).features
        else:
            raise ValueError(f"Unsupported VGG variant: {variant}")

        self.device = device
        self.features = features.eval()
        if device is not None:
            self.features.to(device)
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)

        # Match the original repo's main perceptual/style taps as closely as practical.
        self.layer_map = {
            "conv1_2": 3,
            "conv2_2": 8,
            "conv3_3": 15 if variant == "16" else 17,
            "pool3": 16 if variant == "16" else 18,
        }
        mean = getattr(torch, "tensor")([0.485, 0.456, 0.406], dtype=getattr(torch, "float32")).view(1, 3, 1, 1)
        std = getattr(torch, "tensor")([0.229, 0.224, 0.225], dtype=getattr(torch, "float32")).view(1, 3, 1, 1)
        if device is not None:
            mean = mean.to(device)
            std = std.to(device)
        self.mean = mean
        self.std = std

    def _prepare(self, grayscale_or_rgb: object) -> object:
        torch, _, _ = _require_torch_and_vision()
        x = grayscale_or_rgb
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.clamp(0.0, 1.0)
        return (x - self.mean) / self.std

    def __call__(self, image: object, layers: Sequence[str]) -> dict[str, object]:
        needed = {self.layer_map[name]: name for name in layers}
        outputs: dict[str, object] = {}
        x = self._prepare(image)
        for index, layer in enumerate(self.features):
            x = layer(x)
            name = needed.get(index)
            if name is not None:
                outputs[name] = x
            if len(outputs) == len(layers):
                break
        return outputs


def gram_matrix(features: object) -> object:
    torch, _, _ = _require_torch_and_vision()
    batch, channels, height, width = features.shape
    flattened = features.reshape(batch, channels, height * width)
    gram = getattr(torch, "bmm")(flattened, flattened.transpose(1, 2))
    return gram / max(1, channels * height * width)
