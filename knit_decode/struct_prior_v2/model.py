from __future__ import annotations


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_prior_v2 models. Install with `pip install -e .[train]`.") from error
    return torch, nn


class MultiScaleAutoregressivePrior:
    def __new__(
        cls,
        num_categories: int,
        num_classes: int = 17,
        width: int = 256,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> object:
        torch, nn = _require_torch()

        class _DecoderBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(width)
                self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
                self.norm2 = nn.LayerNorm(width)
                hidden = int(width * mlp_ratio)
                self.mlp = nn.Sequential(
                    nn.Linear(width, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, width),
                )

            def forward(self, x: object, attn_mask: object) -> object:
                h = self.norm1(x)
                attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
                x = x + attn_out
                x = x + self.mlp(self.norm2(x))
                return x

        class _Stage(nn.Module):
            def __init__(self, size: int, cond_channels: int) -> None:
                super().__init__()
                self.size = size
                self.seq_len = size * size
                self.bos_id = num_classes
                self.token_embed = nn.Embedding(num_classes + 1, width)
                self.row_embed = nn.Embedding(size, width)
                self.col_embed = nn.Embedding(size, width)
                self.category_proj = nn.Linear(width, width)
                self.cond_proj = nn.Conv2d(cond_channels, width, kernel_size=1) if cond_channels > 0 else None
                self.blocks = nn.ModuleList([_DecoderBlock() for _ in range(depth)])
                self.norm = nn.LayerNorm(width)
                self.head = nn.Linear(width, num_classes)

            def _causal_mask(self, device: object) -> object:
                mask = getattr(torch, "full")((self.seq_len, self.seq_len), float("-inf"), device=device)
                return getattr(torch, "triu")(mask, diagonal=1)

            def _position_embed(self, batch_size: int, device: object) -> object:
                rows = getattr(torch, "arange")(self.size, device=device)
                cols = getattr(torch, "arange")(self.size, device=device)
                pos = self.row_embed(rows).unsqueeze(1) + self.col_embed(cols).unsqueeze(0)
                pos = pos.reshape(1, self.seq_len, -1)
                return pos.expand(batch_size, -1, -1)

            def forward(self, tokens: object, category_embed: object, cond_map: object | None) -> object:
                batch_size = int(tokens.shape[0])
                x = self.token_embed(tokens)
                x = x + self._position_embed(batch_size, tokens.device) + self.category_proj(category_embed).unsqueeze(1)
                if cond_map is not None and self.cond_proj is not None:
                    cond_feat = self.cond_proj(cond_map).flatten(2).transpose(1, 2)
                    x = x + cond_feat
                mask = self._causal_mask(tokens.device)
                for block in self.blocks:
                    x = block(x, mask)
                return self.head(self.norm(x))

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_classes = num_classes
                self.mask_token_id = num_classes
                self.category_embed = nn.Embedding(num_categories, width)
                self.stage5 = _Stage(size=5, cond_channels=0)
                self.stage10 = _Stage(size=10, cond_channels=num_classes)
                self.stage20 = _Stage(size=20, cond_channels=num_classes)

            def _shift_tokens(self, grid: object) -> object:
                batch_size = int(grid.shape[0])
                flat = grid.reshape(batch_size, -1)
                bos = getattr(torch, "full")((batch_size, 1), self.mask_token_id, device=grid.device, dtype=flat.dtype)
                return getattr(torch, "cat")([bos, flat[:, :-1]], dim=1)

            def _one_hot(self, grid: object, size: int) -> object:
                functional = __import__("importlib").import_module("torch.nn.functional")
                one_hot = functional.one_hot(grid, num_classes=self.num_classes).permute(0, 3, 1, 2).to(dtype=grid.dtype)
                if int(grid.shape[-1]) != size:
                    one_hot = functional.interpolate(one_hot, size=(size, size), mode="nearest")
                return one_hot.to(dtype=getattr(torch, "float32"))

            def forward(self, category_ids: object, grid5: object, grid10: object, grid20: object) -> dict[str, object]:
                category_embed = self.category_embed(category_ids)
                logits5 = self.stage5(self._shift_tokens(grid5), category_embed, None)
                logits5 = logits5.reshape(grid5.shape[0], 5, 5, self.num_classes).permute(0, 3, 1, 2)
                logits10 = self.stage10(self._shift_tokens(grid10), category_embed, self._one_hot(grid5, 10))
                logits10 = logits10.reshape(grid10.shape[0], 10, 10, self.num_classes).permute(0, 3, 1, 2)
                logits20 = self.stage20(self._shift_tokens(grid20), category_embed, self._one_hot(grid10, 20))
                logits20 = logits20.reshape(grid20.shape[0], 20, 20, self.num_classes).permute(0, 3, 1, 2)
                return {"logits5": logits5, "logits10": logits10, "logits20": logits20}

            def _sample_stage(self, stage: object, category_embed: object, size: int, cond_map: object | None, temperature: float) -> object:
                functional = __import__("importlib").import_module("torch.nn.functional")
                batch_size = int(category_embed.shape[0])
                tokens = getattr(torch, "full")((batch_size, size * size), self.mask_token_id, device=category_embed.device, dtype=getattr(torch, "long"))
                for index in range(size * size):
                    logits = stage(tokens, category_embed, cond_map)
                    step_logits = logits[:, index, :] / max(temperature, 1e-6)
                    probs = functional.softmax(step_logits, dim=-1)
                    next_token = getattr(torch, "multinomial")(probs, num_samples=1).squeeze(1)
                    tokens[:, index] = next_token
                return tokens.reshape(batch_size, size, size)

            def sample(self, category_ids: object, temperature: float = 1.0) -> dict[str, object]:
                category_embed = self.category_embed(category_ids)
                grid5 = self._sample_stage(self.stage5, category_embed, 5, None, temperature)
                grid10 = self._sample_stage(self.stage10, category_embed, 10, self._one_hot(grid5, 10), temperature)
                grid20 = self._sample_stage(self.stage20, category_embed, 20, self._one_hot(grid10, 20), temperature)
                return {"pred5": grid5, "pred10": grid10, "pred20": grid20}

        return _Model()
