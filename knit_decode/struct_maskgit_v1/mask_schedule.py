from __future__ import annotations

import math


def schedule(ratio: float, total_unknown: int, method: str = "cosine") -> float:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    total_unknown = max(1, int(total_unknown))
    if method == "uniform":
        mask_ratio = 1.0 - ratio
    elif method.startswith("pow"):
        exponent = float(method.replace("pow", ""))
        mask_ratio = 1.0 - ratio**exponent
    elif method == "cosine":
        mask_ratio = math.cos(math.pi * 0.5 * ratio)
    elif method == "log":
        safe_ratio = max(ratio, 1e-6)
        mask_ratio = -math.log2(safe_ratio) / max(math.log2(total_unknown), 1e-6)
    elif method == "exp":
        mask_ratio = 1.0 - 2 ** (-math.log2(total_unknown) * (1.0 - ratio))
    else:
        raise ValueError(f"Unsupported masking schedule: {method}")
    return max(1e-6, min(1.0, float(mask_ratio)))
