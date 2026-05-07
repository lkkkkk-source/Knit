from __future__ import annotations

import importlib


def _require_torch() -> tuple[object, object]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser models. Install with `pip install -e .[train]`.") from error
    return torch, nn


class KasparTopologyParser:
    """Inverse-Knitting-style grid predictor.

    Input: 1 x 160 x 160 grayscale image
    Output: C x 20 x 20 grid logits
    """

    def __new__(cls, num_classes: int, channels: int = 64, n_res_blocks: int = 6) -> object:
        _, nn = _require_torch()

        class ResidualBlock(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(width, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(width, affine=True),
                )
                self.activation = nn.ReLU(inplace=True)

            def forward(self, x: object) -> object:
                return self.activation(x + self.block(x))

        class _Model(nn.Module):
            def __init__(self, classes: int, width: int, n_blocks: int) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(1, width, kernel_size=3, stride=2, padding=1),   # 160 -> 80
                    nn.InstanceNorm2d(width, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),  # 80 -> 40
                    nn.InstanceNorm2d(width, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),  # 40 -> 20
                    nn.InstanceNorm2d(width, affine=True),
                    nn.ReLU(inplace=True),
                )
                self.res_blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])
                self.head = nn.Sequential(
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(width, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, classes, kernel_size=1),
                )

            def forward(self, x: object) -> object:
                features = self.stem(x)
                features = self.res_blocks(features)
                return self.head(features)

        return _Model(num_classes, channels, n_res_blocks)


def build_parser_model(name: str, num_classes: int) -> object:
    normalized = name.lower()
    if normalized in {"kaspar", "inverse-knitting", "inverse_knitting"}:
        return KasparTopologyParser(num_classes=num_classes)
    raise ValueError(f"Unsupported parser model: {name}")
