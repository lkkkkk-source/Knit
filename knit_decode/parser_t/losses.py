from __future__ import annotations

import importlib
from typing import cast


def _require_torch() -> object:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser training. Install with `pip install -e .[train]`.") from error


def shift_tolerant_cross_entropy(
    logits: object,
    targets: object,
    max_shift: int = 1,
    ignore_index: int = -100,
    weight: object | None = None,
) -> object:
    torch = _require_torch()
    functional = cast(object, importlib.import_module("torch.nn.functional"))
    ce = getattr(functional, "cross_entropy")
    shifts = range(-max_shift, max_shift + 1)
    losses: list[object] = []
    for y_shift in shifts:
        for x_shift in shifts:
            shifted_targets = getattr(torch, "roll")(targets, shifts=(y_shift, x_shift), dims=(-2, -1))
            losses.append(ce(logits, shifted_targets, ignore_index=ignore_index, weight=weight))
    stacked = getattr(torch, "stack")(losses)
    return getattr(torch, "min")(stacked)
