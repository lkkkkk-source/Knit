from __future__ import annotations


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for refiner-v1 models. Install with `pip install -e .[train]`.") from error
    return torch, nn


class RefinerUNet:
    def __new__(cls, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64) -> object:
        _, nn = _require_torch()

        class ConvBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int) -> None:
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_ch, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_ch, affine=True),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x: object) -> object:
                return self.block(x)

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.down1 = ConvBlock(in_channels, base_channels)
                self.pool1 = nn.MaxPool2d(2)
                self.down2 = ConvBlock(base_channels, base_channels * 2)
                self.pool2 = nn.MaxPool2d(2)
                self.mid = ConvBlock(base_channels * 2, base_channels * 4)
                self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
                self.dec1 = ConvBlock(base_channels * 4, base_channels * 2)
                self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
                self.dec2 = ConvBlock(base_channels * 2, base_channels)
                self.out = nn.Sequential(nn.Conv2d(base_channels, out_channels, kernel_size=1), nn.Tanh())

            def forward(self, x: object) -> object:
                skip1 = self.down1(x)
                x = self.pool1(skip1)
                skip2 = self.down2(x)
                x = self.pool2(skip2)
                x = self.mid(x)
                x = self.up1(x)
                x = getattr(__import__("importlib").import_module("torch"), "cat")([x, skip2], dim=1)
                x = self.dec1(x)
                x = self.up2(x)
                x = getattr(__import__("importlib").import_module("torch"), "cat")([x, skip1], dim=1)
                x = self.dec2(x)
                return self.out(x)

        return _Model()
