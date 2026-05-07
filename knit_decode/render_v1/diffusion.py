from __future__ import annotations


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for render-v1 diffusion. Install with `pip install -e .[train]`.") from error


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> object:
    torch = _require_torch()
    return getattr(torch, "linspace")(beta_start, beta_end, num_steps, dtype=getattr(torch, "float32"))


class DiffusionSchedule:
    def __init__(self, num_steps: int) -> None:
        torch = _require_torch()
        self.num_steps = num_steps
        self.betas = linear_beta_schedule(num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = getattr(torch, "cumprod")(self.alphas, dim=0)

    def q_sample(self, x0: object, timesteps: object, noise: object) -> object:
        torch = _require_torch()
        alpha_bars = self.alpha_bars.to(x0.device)
        sqrt_alpha_bar = getattr(torch, "sqrt")(alpha_bars[timesteps]).view(-1, 1, 1, 1)
        sqrt_one_minus = getattr(torch, "sqrt")(1.0 - alpha_bars[timesteps]).view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise
