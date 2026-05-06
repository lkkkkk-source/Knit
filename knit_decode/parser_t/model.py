from __future__ import annotations

import importlib


def _require_torch() -> tuple[object, object]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser models. Install with `pip install -e .[train]`.") from error
    return torch, nn


class TinyTopologyParser:
    """A lightweight parser baseline inspired by dense prediction models."""

    def __new__(cls, num_classes: int) -> object:
        _, nn = _require_torch()

        class _Model(nn.Module):
            def __init__(self, classes: int) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, classes, kernel_size=1),
                )

            def forward(self, x: object) -> object:
                features = self.encoder(x)
                return self.decoder(features)

        return _Model(num_classes)
