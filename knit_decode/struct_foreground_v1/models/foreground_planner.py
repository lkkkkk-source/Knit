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
        spatial_condition_channels: int = 17,
        spatial_hidden_dim: int = 128,
    ) -> object:
        torch, nn = _require_torch()

        class _Planner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.category_embed = nn.Embedding(num_categories, category_embed_dim)
                self.mode_embed = nn.Embedding(max_num_modes, mode_embed_dim)
                self.condition_stem = nn.Sequential(
                    nn.Conv2d(spatial_condition_channels, spatial_hidden_dim // 2, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(spatial_hidden_dim // 2, spatial_hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                self.global_condition_proj = nn.Sequential(
                    nn.Linear(category_embed_dim + mode_embed_dim + 16 + 20 + 20 + 256 + 6 + bbox_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, spatial_hidden_dim),
                    nn.GELU(),
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(spatial_hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                self.local_z_head = nn.Linear(category_embed_dim, max_num_modes)
                self.fg_mask_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
                self.fg_label_head = nn.Conv2d(hidden_dim, 16, kernel_size=1)
                self.decoder_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.bbox_head = nn.Linear(hidden_dim, bbox_dim)
                self.row_proj_head = nn.Linear(hidden_dim, 20)
                self.col_proj_head = nn.Linear(hidden_dim, 20)
                self.grammar_head = nn.Linear(hidden_dim, grammar_dim)
                self.adj_head = nn.Linear(hidden_dim, adjacency_dim)

            def forward(
                self,
                category_ids: object,
                centroid_fg_mask_prob: object,
                centroid_label_prob_16: object,
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
                if centroid_fg_mask_prob is None or centroid_label_prob_16 is None:
                    raise ValueError("ForegroundPlanner requires centroid_fg_mask_prob [B,1,20,20] and centroid_label_prob_16 [B,16,20,20]; got None input.")
                if not hasattr(centroid_fg_mask_prob, "ndim") or not hasattr(centroid_label_prob_16, "ndim"):
                    raise ValueError(
                        "ForegroundPlanner requires centroid_fg_mask_prob [B,1,20,20] and centroid_label_prob_16 [B,16,20,20]; "
                        f"got types {type(centroid_fg_mask_prob)!r} and {type(centroid_label_prob_16)!r}."
                    )
                if int(centroid_fg_mask_prob.ndim) != 4 or int(centroid_label_prob_16.ndim) != 4:
                    raise ValueError(
                        "ForegroundPlanner requires centroid_fg_mask_prob [B,1,20,20] and centroid_label_prob_16 [B,16,20,20]; "
                        f"got shapes {tuple(centroid_fg_mask_prob.shape)} and {tuple(centroid_label_prob_16.shape)}."
                    )
                if tuple(centroid_fg_mask_prob.shape[1:]) != (1, 20, 20) or tuple(centroid_label_prob_16.shape[1:]) != (16, 20, 20):
                    raise ValueError(
                        "ForegroundPlanner requires centroid_fg_mask_prob [B,1,20,20] and centroid_label_prob_16 [B,16,20,20]; "
                        f"got shapes {tuple(centroid_fg_mask_prob.shape)} and {tuple(centroid_label_prob_16.shape)}."
                    )
                if not getattr(centroid_fg_mask_prob.dtype, "is_floating_point", False) or not getattr(centroid_label_prob_16.dtype, "is_floating_point", False):
                    raise ValueError(
                        "ForegroundPlanner requires floating-point centroid priors; "
                        f"got dtypes {centroid_fg_mask_prob.dtype} and {centroid_label_prob_16.dtype}."
                    )
                if int(category_ids.shape[0]) != int(centroid_fg_mask_prob.shape[0]) or int(category_ids.shape[0]) != int(centroid_label_prob_16.shape[0]):
                    raise ValueError(
                        "ForegroundPlanner requires matching batch sizes for category_ids, centroid_fg_mask_prob, and centroid_label_prob_16; "
                        f"got {int(category_ids.shape[0])}, {int(centroid_fg_mask_prob.shape[0])}, and {int(centroid_label_prob_16.shape[0])}."
                    )
                category_embed = self.category_embed(category_ids)
                if centroid_fg_mask_prob.device != category_embed.device or centroid_label_prob_16.device != category_embed.device:
                    raise ValueError(
                        "ForegroundPlanner requires centroid priors to be on the same device as category_ids/model features; "
                        f"got category_embed={category_embed.device}, centroid_fg_mask_prob={centroid_fg_mask_prob.device}, centroid_label_prob_16={centroid_label_prob_16.device}."
                    )
                local_z_logits = self.local_z_head(category_embed)
                if mode_mask is not None:
                    local_z_logits = local_z_logits.masked_fill(mode_mask.logical_not(), float("-inf"))
                if local_z is None:
                    local_z = local_z_logits.argmax(dim=-1)
                mode_embed = self.mode_embed(local_z)
                global_condition = getattr(torch, "cat")(
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
                global_condition_feat = self.global_condition_proj(global_condition).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 20, 20)
                spatial_condition = getattr(torch, "cat")([centroid_fg_mask_prob, centroid_label_prob_16], dim=1)
                spatial_condition_feat = self.condition_stem(spatial_condition)
                decoder_feature = self.decoder(getattr(torch, "cat")([spatial_condition_feat, global_condition_feat], dim=1))
                pooled = self.decoder_pool(decoder_feature).reshape(category_embed.shape[0], -1)
                fg_mask_logits = self.fg_mask_head(decoder_feature)
                fg_label_logits = self.fg_label_head(decoder_feature)
                return {
                    "local_z_logits": local_z_logits,
                    "local_z": local_z,
                    "fg_mask_logits": fg_mask_logits,
                    "fg_label_logits": fg_label_logits,
                    "bbox_pred": self.bbox_head(pooled),
                    "row_projection_pred": self.row_proj_head(pooled),
                    "col_projection_pred": self.col_proj_head(pooled),
                    "grammar_signature_pred": self.grammar_head(pooled),
                    "adjacency_signature_pred": self.adj_head(pooled),
                    "centroid_label_prob_pred": functional.softmax(fg_label_logits, dim=1),
                }

        return _Planner()
