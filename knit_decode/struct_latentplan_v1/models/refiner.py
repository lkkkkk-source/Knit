from __future__ import annotations

from .heads import _require_torch


class PlanConditionedMaskRefiner:
    def __new__(
        cls,
        num_categories: int,
        num_modes: int,
        num_classes: int = 17,
        grid_size: int = 20,
        hidden_dim: int = 384,
        num_layers: int = 8,
        num_heads: int = 6,
        use_2d_rope: bool = True,
    ) -> object:
        torch, nn = _require_torch()

        def _rotate_half(x: object) -> object:
            even = x[..., ::2]
            odd = x[..., 1::2]
            rotated = getattr(torch, "stack")([-odd, even], dim=-1)
            return rotated.flatten(-2)

        def _apply_1d_rope(x: object, positions: object) -> object:
            dim = int(x.shape[-1])
            half = dim // 2
            freq = getattr(torch, "arange")(half, device=x.device, dtype=getattr(torch, "float32"))
            freq = 1.0 / (10000.0 ** (freq / max(1, half)))
            angles = positions.unsqueeze(-1).to(dtype=getattr(torch, "float32")) * freq
            cos = getattr(torch, "cos")(angles).repeat_interleave(2, dim=-1)
            sin = getattr(torch, "sin")(angles).repeat_interleave(2, dim=-1)
            return x * cos + _rotate_half(x) * sin

        def _apply_2d_rope(x: object, size: int) -> object:
            if not use_2d_rope:
                return x
            dim = int(x.shape[-1])
            half = dim // 2
            row_part = x[..., :half]
            col_part = x[..., half:]
            rows = getattr(torch, "arange")(size, device=x.device).unsqueeze(1).expand(size, size).reshape(-1)
            cols = getattr(torch, "arange")(size, device=x.device).unsqueeze(0).expand(size, size).reshape(-1)
            row_part = _apply_1d_rope(row_part, rows)
            col_part = _apply_1d_rope(col_part, cols)
            return getattr(torch, "cat")([row_part, col_part], dim=-1)

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.q_proj = nn.Linear(hidden_dim, hidden_dim)
                self.k_proj = nn.Linear(hidden_dim, hidden_dim)
                self.v_proj = nn.Linear(hidden_dim, hidden_dim)
                self.out_proj = nn.Linear(hidden_dim, hidden_dim)
                self.norm2 = nn.LayerNorm(hidden_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )

            def forward(self, x: object) -> object:
                h = self.norm1(x)
                batch_size, seq_len, _ = h.shape
                head_dim = hidden_dim // num_heads
                q = self.q_proj(h).reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
                k = self.k_proj(h).reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
                v = self.v_proj(h).reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
                q = _apply_2d_rope(q, grid_size)
                k = _apply_2d_rope(k, grid_size)
                scores = getattr(torch, "matmul")(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
                weights = getattr(__import__("importlib").import_module("torch.nn.functional"), "softmax")(scores, dim=-1)
                attn_out = getattr(torch, "matmul")(weights, v).permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)
                x = x + self.out_proj(attn_out)
                x = x + self.mlp(self.norm2(x))
                return x

        class _Refiner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_classes = num_classes
                self.mask_token_id = num_classes
                self.token_embed = nn.Embedding(num_classes + 1, hidden_dim)
                self.row_embed = nn.Embedding(grid_size, hidden_dim)
                self.col_embed = nn.Embedding(grid_size, hidden_dim)
                self.category_embed = nn.Embedding(num_categories, hidden_dim)
                self.mode_embed = nn.Embedding(num_modes, hidden_dim)
                cond_channels = num_classes + 1 + num_classes + 1
                self.plan_proj = nn.Conv2d(cond_channels, hidden_dim, kernel_size=1)
                self.r17_proj = nn.Linear(num_classes, hidden_dim)
                self.fg_ratio_proj = nn.Linear(1, hidden_dim)
                self.blocks = nn.ModuleList([_Block() for _ in range(num_layers)])
                self.norm = nn.LayerNorm(hidden_dim)
                self.occupancy_head = nn.Linear(hidden_dim, 2)
                self.label_head = nn.Linear(hidden_dim, num_classes - 1)

            def _one_hot(self, grid: object, classes: int) -> object:
                functional = __import__("importlib").import_module("torch.nn.functional")
                return functional.one_hot(grid, num_classes=classes).permute(0, 3, 1, 2).to(dtype=getattr(torch, "float32"))

            def _build_condition(self, c5: object, o5: object, r17: object, fg_ratio: object) -> object:
                functional = __import__("importlib").import_module("torch.nn.functional")
                c5_onehot = self._one_hot(c5, self.num_classes)
                c5_up = functional.interpolate(c5_onehot, size=(grid_size, grid_size), mode="nearest")
                o5 = o5.unsqueeze(1).to(dtype=getattr(torch, "float32"))
                o5_up = functional.interpolate(o5, size=(grid_size, grid_size), mode="nearest")
                r17_map = r17.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, grid_size, grid_size)
                fg_ratio_map = fg_ratio.view(-1, 1, 1, 1).expand(-1, 1, grid_size, grid_size)
                return getattr(torch, "cat")([c5_up, o5_up, r17_map, fg_ratio_map], dim=1)

            def forward(
                self,
                tokens20: object,
                category_ids: object,
                z_ids: object,
                c5: object,
                o5: object,
                r17: object,
                fg_ratio: object,
            ) -> dict[str, object]:
                batch_size = int(tokens20.shape[0])
                positions_y = getattr(torch, "arange")(grid_size, device=tokens20.device)
                positions_x = getattr(torch, "arange")(grid_size, device=tokens20.device)
                row_embed = self.row_embed(positions_y).unsqueeze(1).expand(grid_size, grid_size, hidden_dim)
                col_embed = self.col_embed(positions_x).unsqueeze(0).expand(grid_size, grid_size, hidden_dim)
                pos_embed = (row_embed + col_embed).reshape(1, grid_size * grid_size, hidden_dim)
                token_features = self.token_embed(tokens20.reshape(batch_size, -1))
                cond_map = self.plan_proj(self._build_condition(c5, o5, r17, fg_ratio)).flatten(2).transpose(1, 2)
                x = token_features + pos_embed + cond_map
                x = x + self.category_embed(category_ids).unsqueeze(1)
                x = x + self.mode_embed(z_ids).unsqueeze(1)
                x = x + self.r17_proj(r17).unsqueeze(1)
                x = x + self.fg_ratio_proj(fg_ratio.unsqueeze(-1)).unsqueeze(1)
                for block in self.blocks:
                    x = block(x)
                x = self.norm(x)
                occupancy_logits = self.occupancy_head(x).reshape(batch_size, grid_size, grid_size, 2).permute(0, 3, 1, 2)
                label_logits = self.label_head(x).reshape(batch_size, grid_size, grid_size, num_classes - 1).permute(0, 3, 1, 2)
                fg_logit = occupancy_logits[:, 1:2]
                merged = getattr(torch, "cat")([occupancy_logits[:, :1], fg_logit + label_logits], dim=1)
                return {
                    "occupancy_logits": occupancy_logits,
                    "label_logits": label_logits,
                    "merged_logits": merged,
                }

        return _Refiner()
