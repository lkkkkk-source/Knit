from __future__ import annotations

import importlib
from typing import cast


def _require_functional() -> object:
    try:
        return importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser training. Install with `pip install -e .[train]`.") from error


def segmentation_cross_entropy(
    logits: object,
    targets: object,
    ignore_index: int = -100,
    weight: object | None = None,
) -> object:
    functional = cast(object, _require_functional())
    ce = getattr(functional, "cross_entropy")
    return ce(logits, targets, ignore_index=ignore_index, weight=weight)
