from __future__ import annotations

import importlib


def _require_torch() -> tuple[object, object]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser models. Install with `pip install -e .[train]`.") from error
    return torch, nn


def _double_conv(nn: object, in_channels: int, out_channels: int) -> object:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNetTopologyParser:
    """A compact U-Net baseline for simulation-image to topology parsing."""

    def __new__(cls, num_classes: int, base_channels: int = 32) -> object:
        _, nn = _require_torch()

        class _Model(nn.Module):
            def __init__(self, classes: int, width: int) -> None:
                super().__init__()
                self.enc1 = _double_conv(nn, 3, width)
                self.pool1 = nn.MaxPool2d(2)
                self.enc2 = _double_conv(nn, width, width * 2)
                self.pool2 = nn.MaxPool2d(2)
                self.enc3 = _double_conv(nn, width * 2, width * 4)
                self.pool3 = nn.MaxPool2d(2)

                self.bottleneck = _double_conv(nn, width * 4, width * 8)

                self.up3 = nn.ConvTranspose2d(width * 8, width * 4, kernel_size=2, stride=2)
                self.dec3 = _double_conv(nn, width * 8, width * 4)
                self.up2 = nn.ConvTranspose2d(width * 4, width * 2, kernel_size=2, stride=2)
                self.dec2 = _double_conv(nn, width * 4, width * 2)
                self.up1 = nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2)
                self.dec1 = _double_conv(nn, width * 2, width)
                self.head = nn.Conv2d(width, classes, kernel_size=1)

            def forward(self, x: object) -> object:
                enc1 = self.enc1(x)
                enc2 = self.enc2(self.pool1(enc1))
                enc3 = self.enc3(self.pool2(enc2))
                bottleneck = self.bottleneck(self.pool3(enc3))

                up3 = self.up3(bottleneck)
                if up3.shape[-2:] != enc3.shape[-2:]:
                    up3 = nn.functional.interpolate(up3, size=enc3.shape[-2:], mode="bilinear", align_corners=False)
                dec3 = self.dec3(nn.functional.cat([up3, enc3], dim=1))

                up2 = self.up2(dec3)
                if up2.shape[-2:] != enc2.shape[-2:]:
                    up2 = nn.functional.interpolate(up2, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
                dec2 = self.dec2(nn.functional.cat([up2, enc2], dim=1))

                up1 = self.up1(dec2)
                if up1.shape[-2:] != enc1.shape[-2:]:
                    up1 = nn.functional.interpolate(up1, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
                dec1 = self.dec1(nn.functional.cat([up1, enc1], dim=1))
                return self.head(dec1)

        return _Model(num_classes, base_channels)


def build_parser_model(name: str, num_classes: int) -> object:
    normalized = name.lower()
    if normalized == "unet":
        return UNetTopologyParser(num_classes=num_classes)
    raise ValueError(f"Unsupported parser model: {name}")
