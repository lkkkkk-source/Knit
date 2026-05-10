from __future__ import annotations


LAYERNORM_EPSILON = 1e-12


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_maskgit_v1 models. Install with `pip install -e .[train]`.") from error
    return torch, nn


class MultiScaleMaskGitPrior:
    def __new__(
        cls,
        num_categories: int,
        num_classes: int = 17,
        width: int = 512,
        depth: int = 12,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> object:
        torch, nn = _require_torch()

        class _TransformerLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = nn.MultiheadAttention(
                    embed_dim=width,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.attn_dropout = nn.Dropout(dropout)
                self.attn_norm = nn.LayerNorm(width, eps=LAYERNORM_EPSILON)
                hidden = int(width * mlp_ratio)
                self.mlp = nn.Sequential(
                    nn.Linear(width, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, width),
                    nn.Dropout(dropout),
                )
                self.mlp_norm = nn.LayerNorm(width, eps=LAYERNORM_EPSILON)

            def forward(self, x: object) -> object:
                attn_out, _ = self.attn(x, x, x, need_weights=False)
                x = self.attn_norm(x + self.attn_dropout(attn_out))
                x = self.mlp_norm(x + self.mlp(x))
                return x

        class _Stage(nn.Module):
            def __init__(self, size: int, cond_channels: int) -> None:
                super().__init__()
                self.size = size
                self.seq_len = size * size
                self.mask_token_id = num_classes
                self.token_embed = nn.Embedding(num_classes + 1, width)
                self.position_embed = nn.Embedding(self.seq_len, width)
                self.embedding_norm = nn.LayerNorm(width, eps=LAYERNORM_EPSILON)
                self.embedding_dropout = nn.Dropout(dropout)
                self.category_proj = nn.Linear(width, width)
                self.cond_proj = nn.Conv2d(cond_channels, width, kernel_size=1) if cond_channels > 0 else None
                self.layers = nn.ModuleList([_TransformerLayer() for _ in range(depth)])
                self.mlm_dense = nn.Linear(width, width)
                self.mlm_norm = nn.LayerNorm(width, eps=LAYERNORM_EPSILON)
                self.mlm_bias = nn.Parameter(getattr(torch, "zeros")(num_classes))

            def forward(self, tokens: object, category_embed: object, cond_map: object | None) -> object:
                positions = getattr(torch, "arange")(self.seq_len, device=tokens.device).unsqueeze(0)
                x = self.token_embed(tokens) + self.position_embed(positions)
                x = x + self.category_proj(category_embed).unsqueeze(1)
                if cond_map is not None and self.cond_proj is not None:
                    x = x + self.cond_proj(cond_map).flatten(2).transpose(1, 2)
                x = self.embedding_dropout(self.embedding_norm(x))
                for layer in self.layers:
                    x = layer(x)
                x = self.mlm_dense(x)
                functional = __import__("importlib").import_module("torch.nn.functional")
                x = functional.gelu(x)
                x = self.mlm_norm(x)
                logits = getattr(torch, "matmul")(x, self.token_embed.weight[:num_classes].transpose(0, 1)) + self.mlm_bias
                return logits

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_classes = num_classes
                self.mask_token_id = num_classes
                self.category_embed = nn.Embedding(num_categories, width)
                self.stage5 = _Stage(size=5, cond_channels=0)
                self.stage10 = _Stage(size=10, cond_channels=num_classes)
                self.stage20 = _Stage(size=20, cond_channels=num_classes)

            def _one_hot(self, grid: object, size: int) -> object:
                functional = __import__("importlib").import_module("torch.nn.functional")
                one_hot = functional.one_hot(grid, num_classes=self.num_classes).permute(0, 3, 1, 2).to(dtype=getattr(torch, "float32"))
                if int(grid.shape[-1]) != size:
                    one_hot = functional.interpolate(one_hot, size=(size, size), mode="nearest")
                return one_hot

            def _masked_tokens(self, grid: object, mask: object) -> object:
                mask_token = getattr(torch, "full_like")(grid, self.mask_token_id)
                return getattr(torch, "where")(mask, mask_token, grid)

            def forward(
                self,
                category_ids: object,
                grid5: object,
                grid10: object,
                grid20: object,
                mask5: object,
                mask10: object,
                mask20: object,
            ) -> dict[str, object]:
                category_embed = self.category_embed(category_ids)
                logits5 = self.stage5(self._masked_tokens(grid5, mask5).reshape(grid5.shape[0], -1), category_embed, None)
                logits10 = self.stage10(
                    self._masked_tokens(grid10, mask10).reshape(grid10.shape[0], -1),
                    category_embed,
                    self._one_hot(grid5, 10),
                )
                logits20 = self.stage20(
                    self._masked_tokens(grid20, mask20).reshape(grid20.shape[0], -1),
                    category_embed,
                    self._one_hot(grid10, 20),
                )
                return {
                    "logits5": logits5.reshape(grid5.shape[0], 5, 5, self.num_classes).permute(0, 3, 1, 2),
                    "logits10": logits10.reshape(grid10.shape[0], 10, 10, self.num_classes).permute(0, 3, 1, 2),
                    "logits20": logits20.reshape(grid20.shape[0], 20, 20, self.num_classes).permute(0, 3, 1, 2),
                }

            def stage_logits(self, stage_name: str, category_ids: object, tokens: object, cond_grid: object | None = None) -> object:
                category_embed = self.category_embed(category_ids)
                stage = getattr(self, stage_name)
                cond_map = None
                if cond_grid is not None:
                    cond_map = self._one_hot(cond_grid, stage.size)
                return stage(tokens, category_embed, cond_map)

        return _Model()
