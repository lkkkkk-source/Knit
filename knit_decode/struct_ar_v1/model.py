from __future__ import annotations


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_ar_v1 models. Install with `pip install -e .[train]`.") from error
    return torch, nn


class MultiScaleStructureTransformer:
    def __new__(
        cls,
        num_categories: int,
        num_classes: int = 17,
        width: int = 256,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> object:
        _, nn = _require_torch()

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(width)
                self.attn = nn.MultiheadAttention(embed_dim=width, num_heads=heads, batch_first=True)
                self.norm2 = nn.LayerNorm(width)
                hidden = int(width * mlp_ratio)
                self.mlp = nn.Sequential(
                    nn.Linear(width, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, width),
                )

            def forward(self, x: object) -> object:
                h = self.norm1(x)
                attn_out, _ = self.attn(h, h, h, need_weights=False)
                x = x + attn_out
                x = x + self.mlp(self.norm2(x))
                return x

        class _Stage(nn.Module):
            def __init__(self, size: int, extra_channels: int) -> None:
                super().__init__()
                self.size = size
                self.extra_channels = extra_channels
                self.token_embed = nn.Embedding(num_classes + 1, width)
                self.row_embed = nn.Embedding(size, width)
                self.col_embed = nn.Embedding(size, width)
                self.category_proj = nn.Linear(width, width)
                self.input_proj = nn.Conv2d(extra_channels, width, kernel_size=1) if extra_channels > 0 else None
                self.blocks = nn.ModuleList([_Block() for _ in range(depth)])
                self.norm = nn.LayerNorm(width)
                self.head = nn.Linear(width, num_classes)

            def forward(self, tokens: object, category_embed: object, extra: object | None) -> object:
                torch, _ = _require_torch()
                batch_size = int(tokens.shape[0])
                h = self.token_embed(tokens)
                rows = getattr(torch, "arange")(self.size, device=tokens.device)
                cols = getattr(torch, "arange")(self.size, device=tokens.device)
                pos = self.row_embed(rows).unsqueeze(1) + self.col_embed(cols).unsqueeze(0)
                pos = pos.reshape(1, self.size * self.size, -1)
                h = h + pos + self.category_proj(category_embed).unsqueeze(1)
                if extra is not None and self.input_proj is not None:
                    extra_feat = self.input_proj(extra).flatten(2).transpose(1, 2)
                    h = h + extra_feat
                for block in self.blocks:
                    h = block(h)
                return self.head(self.norm(h))

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mask_token_id = num_classes
                self.category_embed = nn.Embedding(num_categories, width)
                self.stage5 = _Stage(size=5, extra_channels=0)
                self.stage10 = _Stage(size=10, extra_channels=num_classes)
                self.stage20 = _Stage(size=20, extra_channels=num_classes)

            def _flatten_tokens(self, grid: object) -> object:
                return grid.reshape(grid.shape[0], -1)

            def _one_hot_grid(self, grid: object, size: int) -> object:
                torch, functional = _require_torch()[0], __import__("importlib").import_module("torch.nn.functional")
                one_hot = functional.one_hot(grid, num_classes=num_classes).permute(0, 3, 1, 2).to(dtype=getattr(torch, "float32"))
                if one_hot.shape[-1] != size:
                    one_hot = functional.interpolate(one_hot, size=(size, size), mode="nearest")
                return one_hot

            def forward(self, category_ids: object, grid5: object, grid10: object, grid20: object, teacher_forcing: bool = True) -> dict[str, object]:
                torch = _require_torch()[0]
                category_embed = self.category_embed(category_ids)
                mask5 = getattr(torch, "full_like")(grid5, self.mask_token_id)
                logits5 = self.stage5(self._flatten_tokens(mask5), category_embed, None)
                logits5 = logits5.reshape(grid5.shape[0], 5, 5, num_classes).permute(0, 3, 1, 2)
                pred5 = getattr(torch, "argmax")(logits5, dim=1)

                source5 = grid5 if teacher_forcing else pred5
                extra10 = self._one_hot_grid(source5, 10)
                mask10 = getattr(torch, "full_like")(grid10, self.mask_token_id)
                logits10 = self.stage10(self._flatten_tokens(mask10), category_embed, extra10)
                logits10 = logits10.reshape(grid10.shape[0], 10, 10, num_classes).permute(0, 3, 1, 2)
                pred10 = getattr(torch, "argmax")(logits10, dim=1)

                source10 = grid10 if teacher_forcing else pred10
                extra20 = self._one_hot_grid(source10, 20)
                mask20 = getattr(torch, "full_like")(grid20, self.mask_token_id)
                logits20 = self.stage20(self._flatten_tokens(mask20), category_embed, extra20)
                logits20 = logits20.reshape(grid20.shape[0], 20, 20, num_classes).permute(0, 3, 1, 2)
                pred20 = getattr(torch, "argmax")(logits20, dim=1)
                return {
                    "logits5": logits5,
                    "logits10": logits10,
                    "logits20": logits20,
                    "pred5": pred5,
                    "pred10": pred10,
                    "pred20": pred20,
                }

        return _Model()
