from __future__ import annotations


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_latentplan_v1 models. Install with `pip install -e .[train]`.") from error
    return torch, nn


class MLP:
    def __new__(cls, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.0) -> object:
        _, nn = _require_torch()
        layers = []
        current = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers.extend([nn.Linear(current, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            current = hidden_dim
        layers.append(nn.Linear(current, output_dim))
        return nn.Sequential(*layers)
