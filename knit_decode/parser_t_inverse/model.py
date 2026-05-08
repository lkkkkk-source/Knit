from __future__ import annotations


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser_t_inverse models. Install with `pip install -e .[train]`.") from error
    return torch, nn


class InverseImg2Prog:
    def __new__(cls, num_classes: int = 17, feat_ch: int = 64, n_res_blocks: int = 6) -> object:
        torch, nn = _require_torch()

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
                self.stem = nn.ModuleList(
                    [
                        nn.Sequential(nn.Conv2d(1, feat_ch, kernel_size=3, stride=2, padding=1), nn.InstanceNorm2d(feat_ch, affine=True), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=2, padding=1), nn.InstanceNorm2d(feat_ch, affine=True), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=2, padding=1), nn.InstanceNorm2d(feat_ch, affine=True), nn.ReLU(inplace=True)),
                    ]
                )
                self.skip_reduce1 = nn.Conv2d(feat_ch * 16, feat_ch, kernel_size=1)
                self.skip_reduce2 = nn.Conv2d(feat_ch * 4, feat_ch, kernel_size=1)
                self.skip_fuse = nn.Sequential(nn.Conv2d(feat_ch * 3, feat_ch, kernel_size=3, padding=1), nn.InstanceNorm2d(feat_ch, affine=True), nn.ReLU(inplace=True))
                self.res_blocks = nn.Sequential(*[ResidualBlock(feat_ch) for _ in range(n_res_blocks)])
                self.head = nn.Sequential(
                    nn.Conv2d(feat_ch * 2, feat_ch, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(feat_ch, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat_ch, num_classes, kernel_size=1),
                )

            def forward(self, x: object) -> object:
                features = []
                for layer in self.stem:
                    x = layer(x)
                    features.append(x)
                act0 = features[-1]
                skip1 = getattr(torch, "pixel_unshuffle")(features[0], 4)
                skip1 = self.skip_reduce1(skip1)
                skip2 = getattr(torch, "pixel_unshuffle")(features[1], 2)
                skip2 = self.skip_reduce2(skip2)
                fused = self.skip_fuse(getattr(torch, "cat")([act0, skip1, skip2], dim=1))
                refined = self.res_blocks(fused)
                logits = self.head(getattr(torch, "cat")([fused, refined], dim=1))
                return logits

        return _Model()
