from __future__ import annotations


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise ImportError("PyTorch is required for refiner_inverse_v1 losses. Install with `pip install -e .[train]`.") from error
    return torch, functional


def gan_hinge_discriminator_loss(real_logits: object, fake_logits: object) -> object:
    torch, functional = _require_torch()
    real_loss = getattr(torch, "relu")(1.0 - real_logits).mean()
    fake_loss = getattr(torch, "relu")(1.0 + fake_logits).mean()
    return real_loss + fake_loss


def gan_hinge_generator_loss(fake_logits: object) -> object:
    return -fake_logits.mean()


def l1_loss(prediction: object, target: object) -> object:
    _, functional = _require_torch()
    return functional.l1_loss(prediction, target)


def parser_cross_entropy(logits: object, targets: object) -> object:
    _, functional = _require_torch()
    return functional.cross_entropy(logits, targets)
