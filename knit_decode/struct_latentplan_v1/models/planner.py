from __future__ import annotations

from .heads import MLP, _require_torch
from ..utils import sample_top_p


class LatentPlanner:
    def __new__(
        cls,
        num_categories: int,
        num_modes: int,
        coarse_size: int = 5,
        num_classes: int = 17,
        category_embed_dim: int = 256,
        mode_embed_dim: int = 256,
        hidden_dim: int = 384,
        num_layers: int = 4,
        coarse_size_10: int = 10,
        grammar_dim: int = 17,
        adjacency_dim: int = 289,
        max_num_modes_per_category: int = 16,
    ) -> object:
        torch, nn = _require_torch()

        class _Planner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_modes = num_modes
                self.max_num_modes_per_category = max_num_modes_per_category
                self.coarse_size = coarse_size
                self.coarse_size_10 = coarse_size_10
                self.num_classes = num_classes
                self.grammar_dim = grammar_dim
                self.adjacency_dim = adjacency_dim
                self.category_embed = nn.Embedding(num_categories, category_embed_dim)
                self.mode_embed = nn.Embedding(max_num_modes_per_category, mode_embed_dim)
                self.z_head = MLP(category_embed_dim, hidden_dim, max_num_modes_per_category, num_layers=max(2, num_layers))
                self.plan_trunk = MLP(category_embed_dim + mode_embed_dim, hidden_dim, hidden_dim, num_layers=max(2, num_layers), dropout=0.1)
                self.c5_head = nn.Linear(hidden_dim, coarse_size * coarse_size * num_classes)
                self.o5_head = nn.Linear(hidden_dim, coarse_size * coarse_size)
                self.c10_head = nn.Linear(hidden_dim, coarse_size_10 * coarse_size_10 * num_classes)
                self.o10_head = nn.Linear(hidden_dim, coarse_size_10 * coarse_size_10)
                self.r17_head = nn.Linear(hidden_dim, num_classes)
                self.fg_ratio_head = nn.Linear(hidden_dim, 1)
                self.row_projection_head = nn.Linear(hidden_dim, 20)
                self.col_projection_head = nn.Linear(hidden_dim, 20)
                self.grammar_signature_head = nn.Linear(hidden_dim, grammar_dim)
                self.adjacency_signature_head = nn.Linear(hidden_dim, adjacency_dim)

            def forward(
                self,
                category_ids: object,
                z_ids: object | None = None,
                mode_mask: object | None = None,
                sample_mode: str = "teacher",
                z_temperature: float = 1.0,
                z_top_p: float = 0.9,
            ) -> dict[str, object]:
                functional = __import__("importlib").import_module("torch.nn.functional")
                category_embed = self.category_embed(category_ids)
                z_logits = self.z_head(category_embed)
                if mode_mask is not None:
                    z_logits = z_logits.masked_fill(mode_mask.logical_not(), float("-inf"))
                if z_ids is None:
                    if sample_mode == "argmax":
                        z_ids = z_logits.argmax(dim=-1)
                    else:
                        z_ids = sample_top_p(z_logits, temperature=z_temperature, top_p=z_top_p)
                mode_embed = self.mode_embed(z_ids)
                hidden = self.plan_trunk(getattr(torch, "cat")([category_embed, mode_embed], dim=-1))
                c5_logits = self.c5_head(hidden).reshape(-1, coarse_size, coarse_size, num_classes).permute(0, 3, 1, 2)
                o5_logits = self.o5_head(hidden).reshape(-1, 1, coarse_size, coarse_size)
                c10_logits = self.c10_head(hidden).reshape(-1, coarse_size_10, coarse_size_10, num_classes).permute(0, 3, 1, 2)
                o10_logits = self.o10_head(hidden).reshape(-1, 1, coarse_size_10, coarse_size_10)
                r17_pred = functional.softmax(self.r17_head(hidden), dim=-1)
                fg_ratio_pred = getattr(torch, "sigmoid")(self.fg_ratio_head(hidden).squeeze(-1))
                row_projection_pred = getattr(torch, "sigmoid")(self.row_projection_head(hidden))
                col_projection_pred = getattr(torch, "sigmoid")(self.col_projection_head(hidden))
                grammar_signature_pred = self.grammar_signature_head(hidden)
                adjacency_signature_pred = self.adjacency_signature_head(hidden)
                z_logprob = functional.log_softmax(z_logits, dim=-1).gather(-1, z_ids.unsqueeze(-1)).squeeze(-1)
                return {
                    "z_logits": z_logits,
                    "z_ids": z_ids,
                    "z_logprob": z_logprob,
                    "c5_logits": c5_logits,
                    "o5_logits": o5_logits,
                    "c10_logits": c10_logits,
                    "o10_logits": o10_logits,
                    "r17_pred": r17_pred,
                    "fg_ratio_pred": fg_ratio_pred,
                    "row_projection_pred": row_projection_pred,
                    "col_projection_pred": col_projection_pred,
                    "grammar_signature_pred": grammar_signature_pred,
                    "adjacency_signature_pred": adjacency_signature_pred,
                }

        return _Planner()
