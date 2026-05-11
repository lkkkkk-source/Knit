from __future__ import annotations


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_foreground_v1 planner.") from error
    return torch, nn


class ForegroundCanonicalPlanner:
    def __new__(
        cls,
        num_categories: int,
        max_num_modes: int,
        hidden_dim: int = 384,
        category_embed_dim: int = 256,
        mode_embed_dim: int = 256,
        grammar_dim: int = 16,
        adjacency_dim: int = 256,
        bbox_dim: int = 10,
    ) -> object:
        torch, nn = _require_torch()

        class _Planner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.category_embed = nn.Embedding(num_categories, category_embed_dim)
                self.mode_embed = nn.Embedding(max_num_modes, mode_embed_dim)
                self.trunk = nn.Sequential(
                    nn.Linear(category_embed_dim + mode_embed_dim + 16 + 20 + 20 + 256 + 6 + bbox_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                self.local_z_head = nn.Linear(category_embed_dim, max_num_modes)
                self.fg_mask_head = nn.Linear(hidden_dim, 20 * 20)
                self.fg_label_head = nn.Linear(hidden_dim, 16 * 20 * 20)
                self.bbox_head = nn.Linear(hidden_dim, bbox_dim)
                self.row_proj_head = nn.Linear(hidden_dim, 20)
                self.col_proj_head = nn.Linear(hidden_dim, 20)
                self.grammar_head = nn.Linear(hidden_dim, grammar_dim)
                self.adj_head = nn.Linear(hidden_dim, adjacency_dim)

            def forward(
                self,
                category_ids: object,
                centroid_label_hist: object,
                centroid_row_projection: object,
                centroid_col_projection: object,
                centroid_adjacency: object,
                centroid_transition_stats: object,
                centroid_bbox_stats: object,
                local_z: object | None = None,
                mode_mask: object | None = None,
            ) -> dict[str, object]:
                functional = __import__("importlib").import_module("torch.nn.functional")
                category_embed = self.category_embed(category_ids)
                local_z_logits = self.local_z_head(category_embed)
                if mode_mask is not None:
                    local_z_logits = local_z_logits.masked_fill(mode_mask.logical_not(), float("-inf"))
                if local_z is None:
                    local_z = local_z_logits.argmax(dim=-1)
                mode_embed = self.mode_embed(local_z)
                cond = getattr(torch, "cat")(
                    [
                        category_embed,
                        mode_embed,
                        centroid_label_hist,
                        centroid_row_projection,
                        centroid_col_projection,
                        centroid_adjacency,
                        centroid_transition_stats,
                        centroid_bbox_stats,
                    ],
                    dim=-1,
                )
                hidden = self.trunk(cond)
                fg_mask_logits = self.fg_mask_head(hidden).reshape(-1, 1, 20, 20)
                fg_label_logits = self.fg_label_head(hidden).reshape(-1, 16, 20, 20)
                return {
                    "local_z_logits": local_z_logits,
                    "local_z": local_z,
                    "fg_mask_logits": fg_mask_logits,
                    "fg_label_logits": fg_label_logits,
                    "bbox_pred": self.bbox_head(hidden),
                    "row_projection_pred": self.row_proj_head(hidden),
                    "col_projection_pred": self.col_proj_head(hidden),
                    "grammar_signature_pred": self.grammar_head(hidden),
                    "adjacency_signature_pred": self.adj_head(hidden),
                }

        return _Planner()
