from __future__ import annotations

from .mask_schedule import schedule


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_maskgit_v1 decoding. Install with `pip install -e .[train]`.") from error


def mask_by_random_topk(mask_len: object, probs: object, temperature: float = 1.0) -> object:
    torch = _require_torch()
    safe_probs = probs.clamp_min(1e-9)
    confidence = safe_probs.log() + float(temperature) * (-(-(getattr(torch, "rand_like")(safe_probs).clamp_min(1e-9)).log()).log())
    batch_size, seq_len = confidence.shape
    masking = getattr(torch, "zeros")((batch_size, seq_len), dtype=getattr(torch, "bool"), device=confidence.device)
    for batch_index in range(batch_size):
        k = int(mask_len[batch_index].item())
        if k <= 0:
            continue
        indices = getattr(torch, "topk")(confidence[batch_index], k=min(k, seq_len), largest=False).indices
        masking[batch_index, indices] = True
    return masking


def decode(
    inputs: object,
    tokens_to_logits: object,
    mask_token_id: int,
    num_iter: int,
    choice_temperature: float = 4.5,
    mask_scheduling_method: str = "cosine",
) -> object:
    torch = _require_torch()
    functional = __import__("importlib").import_module("torch.nn.functional")
    cur_ids = inputs.to(dtype=getattr(torch, "long"))
    unknown_number_in_the_beginning = cur_ids.eq(mask_token_id).sum(dim=-1)
    for step in range(max(1, int(num_iter))):
        logits = tokens_to_logits(cur_ids)
        sampled_ids = getattr(torch, "distributions").Categorical(logits=logits).sample()
        unknown_map = cur_ids.eq(mask_token_id)
        sampled_ids = getattr(torch, "where")(unknown_map, sampled_ids, cur_ids)

        probs = functional.softmax(logits, dim=-1)
        selected_probs = probs.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
        inf_tensor = getattr(torch, "full_like")(selected_probs, float("inf"))
        selected_probs = getattr(torch, "where")(unknown_map, selected_probs, inf_tensor)

        ratio = float(step + 1) / float(max(1, int(num_iter)))
        mask_ratio = schedule(ratio, int(unknown_number_in_the_beginning.max().item()), method=mask_scheduling_method)
        desired_mask_len = getattr(torch, "floor")(unknown_number_in_the_beginning.to(dtype=getattr(torch, "float32")) * mask_ratio).to(dtype=getattr(torch, "long"))
        current_unknown = unknown_map.sum(dim=-1)
        max_mask_len = (current_unknown - 1).clamp_min(0)
        mask_len = getattr(torch, "minimum")(desired_mask_len, max_mask_len)
        if int(mask_len.max().item()) <= 0:
            cur_ids = sampled_ids
            continue
        masking = mask_by_random_topk(mask_len, selected_probs, temperature=float(choice_temperature) * (1.0 - ratio))
        cur_ids = getattr(torch, "where")(masking, getattr(torch, "full_like")(sampled_ids, mask_token_id), sampled_ids)
    return cur_ids
