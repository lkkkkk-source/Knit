from __future__ import annotations

import csv
from pathlib import Path


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser_t_inverse losses. Install with `pip install -e .[train]`.") from error


def weighted_cross_entropy(logits: object, targets: object, weight: object | None = None) -> object:
    torch = _require_torch()
    functional = __import__("importlib").import_module("torch.nn.functional")
    return functional.cross_entropy(logits, targets, weight=weight)


def build_syntax_penalties(syntax_dir: str | Path, num_classes: int = 17) -> list[object]:
    torch = _require_torch()
    syntax_dir = Path(syntax_dir)
    penalties: list[object] = []
    for index in range(1, 9):
        matrix_path = syntax_dir / f"T{index}.txt"
        rows: list[list[float]] = []
        with matrix_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                rows.append([float(value) for value in row])
        if len(rows) != num_classes or any(len(row) != num_classes for row in rows):
            raise ValueError(f"Unexpected syntax matrix shape in {matrix_path}")
        matrix = getattr(torch, "tensor")(rows, dtype=getattr(torch, "float32"))
        matrix = 1.0 - getattr(torch, "clamp")(matrix, 0.0, 1.0)
        penalties.append(matrix)
    return penalties


def syntax_loss(logits: object, penalties: list[object], syntax_softmax: bool = True) -> object:
    torch = _require_torch()
    probs = getattr(__import__("importlib").import_module("torch.nn.functional"), "softmax")(logits, dim=1) if syntax_softmax else logits
    probs = probs.permute(0, 2, 3, 1)
    offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    losses = []
    for (dx, dy), penalty in zip(offsets, penalties):
        src_y0 = 1 if dy < 0 else 0
        src_y1 = probs.shape[1] - 1 if dy > 0 else probs.shape[1]
        src_x0 = 1 if dx < 0 else 0
        src_x1 = probs.shape[2] - 1 if dx > 0 else probs.shape[2]
        trg_y0 = 0 if dy < 0 else 1 if dy > 0 else 0
        trg_y1 = probs.shape[1] - 1 if dy < 0 else probs.shape[1] if dy == 0 else probs.shape[1]
        trg_x0 = 0 if dx < 0 else 1 if dx > 0 else 0
        trg_x1 = probs.shape[2] - 1 if dx < 0 else probs.shape[2] if dx == 0 else probs.shape[2]
        src = probs[:, src_y0:src_y1, src_x0:src_x1, :]
        trg = probs[:, trg_y0:trg_y1, trg_x0:trg_x1, :]
        penalty = penalty.to(logits.device)
        pair_loss = getattr(torch, "einsum")("bhwi,ij,bhwj->bhw", src, penalty, trg)
        losses.append(getattr(torch, "sqrt")(pair_loss.pow(2) + 1e-8).mean())
    return sum(losses) / max(1, len(losses))
