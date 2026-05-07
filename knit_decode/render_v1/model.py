from __future__ import annotations

import math


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for render-v1 models. Install with `pip install -e .[train]`.") from error
    return torch, nn


class SinusoidalTimeEmbedding:
    def __new__(cls, dim: int) -> object:
        torch, nn = _require_torch()

        class _Module(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.width = width

            def forward(self, timesteps: object) -> object:
                half = self.width // 2
                device = timesteps.device
                factor = math.log(10000.0) / max(1, half - 1)
                exponent = getattr(torch, "arange")(half, device=device, dtype=getattr(torch, "float32"))
                exponent = getattr(torch, "exp")(-factor * exponent)
                args = timesteps.float().unsqueeze(1) * exponent.unsqueeze(0)
                emb = getattr(torch, "cat")([getattr(torch, "sin")(args), getattr(torch, "cos")(args)], dim=1)
                if self.width % 2 == 1:
                    zeros = getattr(torch, "zeros")((emb.shape[0], 1), device=device, dtype=emb.dtype)
                    emb = getattr(torch, "cat")([emb, zeros], dim=1)
                return emb

        return _Module(dim)


class CategoryConditionalUNet:
    def __new__(cls, num_categories: int, image_channels: int = 3, base_channels: int = 64, time_dim: int = 128) -> object:
        _, nn = _require_torch()

        class ResidualBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.norm1 = nn.GroupNorm(8, in_channels)
                self.act1 = nn.SiLU()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                self.norm2 = nn.GroupNorm(8, out_channels)
                self.act2 = nn.SiLU()
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
                self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

            def forward(self, x: object, cond: object) -> object:
                residual = self.skip(x)
                h = self.conv1(self.act1(self.norm1(x)))
                scale_shift = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
                scale, shift = scale_shift.chunk(2, dim=1)
                h = self.norm2(h)
                h = h * (1 + scale) + shift
                h = self.conv2(self.act2(h))
                return h + residual

        class DownBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
                super().__init__()
                self.res1 = ResidualBlock(in_channels, out_channels, cond_dim)
                self.res2 = ResidualBlock(out_channels, out_channels, cond_dim)
                self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

            def forward(self, x: object, cond: object) -> tuple[object, object]:
                x = self.res1(x, cond)
                x = self.res2(x, cond)
                skip = x
                x = self.down(x)
                return x, skip

        class UpBlock(nn.Module):
            def __init__(self, in_channels: int, skip_channels: int, out_channels: int, cond_dim: int) -> None:
                super().__init__()
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
                self.res1 = ResidualBlock(out_channels + skip_channels, out_channels, cond_dim)
                self.res2 = ResidualBlock(out_channels, out_channels, cond_dim)

            def forward(self, x: object, skip: object, cond: object) -> object:
                x = self.up(x)
                x = getattr(__import__("importlib").import_module("torch"), "cat")([x, skip], dim=1)
                x = self.res1(x, cond)
                x = self.res2(x, cond)
                return x

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.time_embed = SinusoidalTimeEmbedding(time_dim)
                self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
                self.category_embed = nn.Embedding(num_categories, time_dim)
                self.in_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
                self.down1 = DownBlock(base_channels, base_channels, time_dim)
                self.down2 = DownBlock(base_channels, base_channels * 2, time_dim)
                self.mid1 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
                self.mid2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
                self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_dim)
                self.up2 = UpBlock(base_channels, base_channels, base_channels, time_dim)
                self.out = nn.Sequential(
                    nn.GroupNorm(8, base_channels),
                    nn.SiLU(),
                    nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1),
                )

            def forward(self, x: object, timesteps: object, category_ids: object) -> object:
                cond = self.time_mlp(self.time_embed(timesteps)) + self.category_embed(category_ids)
                x = self.in_conv(x)
                x, skip1 = self.down1(x, cond)
                x, skip2 = self.down2(x, cond)
                x = self.mid1(x, cond)
                x = self.mid2(x, cond)
                x = self.up1(x, skip2, cond)
                x = self.up2(x, skip1, cond)
                return self.out(x)

        return _Model()
