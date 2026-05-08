from __future__ import annotations


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise ImportError("PyTorch is required for inverse_residual_v1 losses. Install with `pip install -e .[train]`.") from error
    return torch, functional


def gan_hinge_discriminator_loss(real_logits: object, fake_logits: object) -> object:
    torch, _ = _require_torch()
    return getattr(torch, "relu")(1.0 - real_logits).mean() + getattr(torch, "relu")(1.0 + fake_logits).mean()


def gan_hinge_generator_loss(fake_logits: object) -> object:
    return -fake_logits.mean()


def l1_loss(prediction: object, target: object) -> object:
    _, functional = _require_torch()
    return functional.l1_loss(prediction, target)


def cross_entropy_loss(logits: object, targets: object) -> object:
    _, functional = _require_torch()
    return functional.cross_entropy(logits, targets)
