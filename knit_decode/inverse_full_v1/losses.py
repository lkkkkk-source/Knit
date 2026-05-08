from __future__ import annotations

from knit_decode.inverse_residual_v1.losses import (
    cross_entropy_loss,
    gan_hinge_discriminator_loss,
    gan_hinge_generator_loss,
    l1_loss,
)


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise ImportError("PyTorch is required for inverse_full_v1 losses. Install with `pip install -e .[train]`.") from error
    return torch, functional


def mil_cross_entropy_loss(logits: object, targets: object) -> object:
    torch, functional = _require_torch()
    offsets = [(0, 0), (-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    losses = []
    for dx, dy in offsets:
        shifted = getattr(torch, "roll")(targets, shifts=(dy, dx), dims=(1, 2))
        ce = functional.cross_entropy(logits, shifted, reduction="none")
        losses.append(ce.mean(dim=(1, 2)))
    stacked = getattr(torch, "stack")(losses, dim=1)
    return stacked.min(dim=1).values.mean()


__all__ = [
    "cross_entropy_loss",
    "gan_hinge_discriminator_loss",
    "gan_hinge_generator_loss",
    "l1_loss",
    "mil_cross_entropy_loss",
]
