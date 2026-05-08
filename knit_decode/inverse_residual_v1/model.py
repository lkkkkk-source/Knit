from __future__ import annotations


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for inverse_residual_v1 models. Install with `pip install -e .[train]`.") from error
    return torch, nn


class ResidualRefiner:
    def __new__(cls, feat_ch: int = 64, n_res_blocks: int = 6) -> object:
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
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, feat_ch, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(feat_ch, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(feat_ch, affine=True),
                    nn.ReLU(inplace=True),
                )
                self.body = nn.Sequential(*[ResidualBlock(feat_ch) for _ in range(n_res_blocks)])
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(feat_ch * 2, feat_ch, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(feat_ch, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(feat_ch, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat_ch, 1, kernel_size=1),
                    nn.Tanh(),
                )

            def forward(self, x: object) -> object:
                feat = self.encoder(x)
                body = self.body(feat)
                return self.decoder(getattr(__import__("importlib").import_module("torch"), "cat")([feat, body], dim=1)) * 0.5

        return _Model()


class ResidualConditionalPatchDiscriminator:
    def __new__(cls, image_channels: int = 1, num_classes: int = 17, feat_ch: int = 64) -> object:
        _, nn = _require_torch()

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Conv2d(num_classes, 4, kernel_size=1)
                self.net = nn.Sequential(
                    nn.Conv2d(image_channels + 4, feat_ch, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(feat_ch, feat_ch * 2, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(feat_ch * 2, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(feat_ch * 2, feat_ch * 4, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(feat_ch * 4, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(feat_ch * 4, feat_ch * 8, kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(feat_ch * 8, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(feat_ch * 8, 1, kernel_size=3, stride=1, padding=1),
                )

            def forward(self, image: object, instruction_onehot: object) -> object:
                import importlib

                functional = importlib.import_module("torch.nn.functional")
                cond = functional.interpolate(instruction_onehot, size=image.shape[-2:], mode="nearest")
                cond = self.embed(cond)
                x = getattr(importlib.import_module("torch"), "cat")([image, cond], dim=1)
                return self.net(x)

        return _Model()
