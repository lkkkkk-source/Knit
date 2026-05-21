from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypedDict, cast

from .utils import IGNORE_INDEX, EXPECTED_DESCRIPTOR_DIM, InstructionGrammarPriorError, bbox_from_mask, bbox_vector, foreground_descriptor, require_centroid_sketch_fields, require_ignore_index, resolve_canonical_mode, to_plain_list, validate_foreground_labels, validate_instruction_grammar_prior_category


class ForegroundSample(TypedDict):
    sample_id: str
    category: str
    input_path: str
    target_path: str
    index_path: str


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        data = importlib.import_module("torch.utils.data")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_foreground_v1 dataset.") from error
    return torch, data


def load_manifest(path: str | Path) -> list[ForegroundSample]:
    rows: list[ForegroundSample] = []
    path = Path(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        row = {}
        for key in ["sample_id", "category", "input_path", "target_path", "index_path"]:
            value = payload.get(key)
            if not isinstance(value, str):
                raise ValueError(f"Missing {key!r} in manifest row: {payload!r}")
            row[key] = value
        rows.append(cast(ForegroundSample, row))
    return rows


class ForegroundDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        cache_path: str | Path,
        category_to_id: dict[str, int] | None = None,
        *,
        exclude_unseen_categories: bool = False,
        prior_source: str = "category_kmeans",
        prior_mode_sampling: str = "sample",
        prior_label_prob_key: str = "basis_label_prob_16",
        mode_prior_key: str = "mode_prior_smoothed",
        fallback_prior_source: str = "category_kmeans",
        fallback_if_missing: bool = True,
        prior_seed: int = 0,
    ) -> None:
        torch, _ = _require_torch()
        self.manifest_path = Path(manifest_path)
        self.samples = load_manifest(self.manifest_path)
        self.cache_payload = getattr(torch, "load")(Path(cache_path), map_location="cpu")
        self.cache_by_id = {entry["sample_id"]: entry for entry in self.cache_payload["items"]}
        self.exclude_unseen_categories = bool(exclude_unseen_categories)
        self.skipped_unseen_count = 0
        self.skipped_unseen_categories: list[str] = []
        self.prior_source = str(prior_source)
        self.prior_mode_sampling = str(prior_mode_sampling)
        self.prior_label_prob_key = str(prior_label_prob_key)
        self.mode_prior_key = str(mode_prior_key)
        self.fallback_prior_source = str(fallback_prior_source)
        self.fallback_if_missing = bool(fallback_if_missing)
        self.prior_seed = int(prior_seed)
        self.prior_source_counts: dict[str, int] = {}
        self.fallback_count = 0
        self.fallback_reasons: dict[str, int] = {}
        self.prior_warnings: dict[str, int] = {}
        self.prior_mode_hist_by_category: dict[str, dict[str, int]] = {}
        categories = sorted({sample["category"] for sample in self.samples})
        self.category_to_id = category_to_id or {category: index for index, category in enumerate(categories)}
        planner_cf = self.cache_payload.get("config", {}).get("planner", {})
        data_cf = self.cache_payload.get("config", {}).get("data", {})
        self.canonical_mode = resolve_canonical_mode(data_cf)
        self.ignore_index = require_ignore_index(data_cf)
        self.max_num_modes_per_category = int(planner_cf.get("max_num_modes_per_category", planner_cf.get("num_modes_per_category", 16)))
        self._validate_alignment()
        require_centroid_sketch_fields(self.cache_payload, context="Foreground dataset cache")
        if self.exclude_unseen_categories:
            filtered = []
            skipped_categories = set()
            for sample in self.samples:
                cached = self.cache_by_id[sample["sample_id"]]
                if bool(cached.get("is_unseen_category", False)):
                    self.skipped_unseen_count += 1
                    skipped_categories.add(str(sample["category"]))
                    continue
                filtered.append(sample)
            self.samples = filtered
            self.skipped_unseen_categories = sorted(skipped_categories)

    def _bump_prior_stats(self, category: str, source: str, mode_id: int, fallback_reason: str | None = None) -> None:
        self.prior_source_counts[source] = int(self.prior_source_counts.get(source, 0)) + 1
        hist = self.prior_mode_hist_by_category.setdefault(category, {})
        hist[str(mode_id)] = int(hist.get(str(mode_id), 0)) + 1
        if fallback_reason is not None:
            self.fallback_count += 1
            self.fallback_reasons[fallback_reason] = int(self.fallback_reasons.get(fallback_reason, 0)) + 1

    def _bump_prior_warning(self, warning: str) -> None:
        self.prior_warnings[warning] = int(self.prior_warnings.get(warning, 0)) + 1

    def prior_debug_summary(self) -> dict[str, object]:
        return {
            "prior_source": self.prior_source,
            "prior_mode_sampling": self.prior_mode_sampling,
            "prior_label_prob_key": self.prior_label_prob_key,
            "mode_prior_key": self.mode_prior_key,
            "fallback_prior_source": self.fallback_prior_source,
            "fallback_if_missing": bool(self.fallback_if_missing),
            "prior_source_counts": dict(sorted(self.prior_source_counts.items())),
            "fallback_count": int(self.fallback_count),
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "warnings": dict(sorted(self.prior_warnings.items())),
            "mode_id_histogram_by_category": self.prior_mode_hist_by_category,
        }

    def _choose_mode(self, category: str, priors: list[float], mode_count: int, sample_id: str) -> int:
        if mode_count <= 0:
            return 0
        strategy = self.prior_mode_sampling
        if strategy in {"top", "deterministic"}:
            return int(max(range(mode_count), key=lambda index: float(priors[index]) if index < len(priors) else 0.0))
        rng = random.Random(f"{self.prior_seed}:{category}:{sample_id}")
        weights = [max(0.0, float(priors[index]) if index < len(priors) else 0.0) for index in range(mode_count)]
        total = sum(weights)
        if total <= 0.0:
            return int(rng.randrange(mode_count))
        draw = rng.random() * total
        acc = 0.0
        for index, weight in enumerate(weights):
            acc += weight
            if draw <= acc:
                return int(index)
        return int(mode_count - 1)

    def _binary_mask_from_prob(self, fg_prob: list[list[list[float]]]) -> list[list[int]]:
        grid = fg_prob[0] if fg_prob and isinstance(fg_prob[0], list) else fg_prob
        mask = [[1 if float(grid[y_pos][x_pos]) >= 0.5 else 0 for x_pos in range(20)] for y_pos in range(20)]
        if sum(sum(row) for row in mask) <= 0:
            flat = sorted((float(grid[y][x]), y, x) for y in range(20) for x in range(20))
            for _, y_pos, x_pos in flat[-max(1, len(flat) // 20):]:
                mask[y_pos][x_pos] = 1
        return mask

    def _argmax_labels(self, label_prob: list[list[list[float]]], mask: list[list[int]]) -> list[list[int]]:
        labels: list[list[int]] = []
        for y_pos in range(20):
            row: list[int] = []
            for x_pos in range(20):
                if not mask[y_pos][x_pos]:
                    row.append(0)
                    continue
                label_index = max(range(16), key=lambda idx: float(label_prob[idx][y_pos][x_pos]))
                row.append(int(label_index + 1))
            labels.append(row)
        return labels

    def _instruction_grammar_prior(self, category: str, sample_id: str) -> dict[str, object] | None:
        validated = validate_instruction_grammar_prior_category(
            self.cache_payload,
            category,
            label_prob_key=self.prior_label_prob_key,
            mode_prior_key=self.mode_prior_key,
        )
        for warning in cast(list[str], validated["warnings"]):
            self._bump_prior_warning(warning)
        fg_modes = cast(list[object], validated["fg_modes"])
        label_modes = cast(list[object], validated["label_modes"])
        priors = cast(list[float], validated["mode_prior"])
        mode_id = self._choose_mode(category, priors, len(fg_modes), sample_id)
        fg_prob = to_plain_list(fg_modes[mode_id])
        label_prob = to_plain_list(label_modes[mode_id])
        if not isinstance(fg_prob, list) or not isinstance(label_prob, list):
            raise InstructionGrammarPriorError(
                "Invalid instruction_matrix_grammar_prior: expected schema "
                "'foreground_v1_instruction_matrix_grammar_prior_v1'; selected mode tensors must be lists."
            )
        fg_mask_bin = self._binary_mask_from_prob(cast(list[list[list[float]]], fg_prob))
        argmax_labels = self._argmax_labels(cast(list[list[list[float]]], label_prob), fg_mask_bin)
        desc = foreground_descriptor(argmax_labels, fg_mask_bin, bbox_from_mask([[bool(value) for value in row] for row in fg_mask_bin]))
        entry = cast(dict[str, object], validated["entry"])
        label_hist = to_plain_list(entry.get("category_label_hist_16", desc["label_hist_16"]))
        return {
            "source": "instruction_matrix_grammar",
            "mode_id": int(mode_id),
            "num_modes": int(validated["num_modes"]),
            "fg_mask_prob": fg_prob,
            "fg_mask_bin": fg_mask_bin,
            "label_prob_16": label_prob,
            "label_hist": label_hist if isinstance(label_hist, list) and len(label_hist) == 16 else desc["label_hist_16"],
            "row_projection": desc["row_projection"],
            "col_projection": desc["col_projection"],
            "adjacency": desc["adjacency_signature"],
            "transition_stats": desc["transition_2x2_stats"],
            "bbox_stats": bbox_vector(bbox_from_mask([[bool(value) for value in row] for row in fg_mask_bin])),
        }

    def _centroid_prior(self, category: str, local_z: int) -> dict[str, object]:
        centroid = self.cache_payload["centroid_sketch_by_category"].get(category, {}).get(local_z, {})
        if not centroid:
            raise ValueError(f"Missing centroid sketch for category={category!r} local_z={local_z}.")
        if "centroid_fg_mask_prob" not in centroid or "centroid_fg_mask" not in centroid or "centroid_label_prob_16" not in centroid:
            raise ValueError(
                f"Centroid sketch for category={category!r} local_z={local_z} "
                "is missing centroid_fg_mask_prob / centroid_fg_mask / centroid_label_prob_16. Please rebuild the cache."
            )
        return {
            "source": "category_kmeans",
            "mode_id": int(local_z),
            "fg_mask_prob": to_plain_list(centroid.get("centroid_fg_mask_prob")),
            "fg_mask_bin": to_plain_list(centroid.get("centroid_fg_mask")),
            "label_prob_16": to_plain_list(centroid.get("centroid_label_prob_16")),
            "label_hist": centroid.get("centroid_label_hist", [0.0] * 16),
            "row_projection": centroid.get("centroid_row_projection", [0.0] * 20),
            "col_projection": centroid.get("centroid_col_projection", [0.0] * 20),
            "adjacency": centroid.get("centroid_adjacency", [0.0] * 256),
            "transition_stats": centroid.get("centroid_transition_stats", [0.0] * 6),
            "bbox_stats": centroid.get("centroid_bbox_stats", [0.0] * 10),
        }

    def _validate_alignment(self) -> None:
        manifest_ids = {sample["sample_id"] for sample in self.samples}
        cache_ids = set(self.cache_by_id)
        if manifest_ids != cache_ids:
            missing_in_cache = sorted(manifest_ids - cache_ids)
            missing_in_manifest = sorted(cache_ids - manifest_ids)
            raise ValueError(
                f"Manifest/cache sample_id mismatch in foreground dataset: "
                f"missing_in_cache={missing_in_cache[:5]} missing_in_manifest={missing_in_manifest[:5]}"
            )
        for sample in self.samples:
            cached = self.cache_by_id[sample["sample_id"]]
            if cached["category"] != sample["category"]:
                raise ValueError(f"Category mismatch for sample_id={sample['sample_id']}")
            for key in ("input_path", "target_path", "index_path"):
                if str(cached.get(key)) != str(sample[key]):
                    raise ValueError(f"Path mismatch for sample_id={sample['sample_id']} field={key}")
            validate_foreground_labels(cached["fg_y20"], cached["fg_mask20"], context=f"dataset[{sample['sample_id']}]")
            if cached.get("canonical_mode", self.canonical_mode) != self.canonical_mode:
                raise ValueError(
                    f"Canonical mode mismatch for sample_id={sample['sample_id']}: "
                    f"cache has {cached.get('canonical_mode')!r}, dataset expects {self.canonical_mode!r}"
                )
            descriptor = cached.get("descriptor")
            if not isinstance(descriptor, list) or len(descriptor) != EXPECTED_DESCRIPTOR_DIM:
                raise ValueError(
                    f"Descriptor mismatch for sample_id={sample['sample_id']}: expected dim {EXPECTED_DESCRIPTOR_DIM}, got {len(descriptor) if isinstance(descriptor, list) else type(descriptor)}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        cached = self.cache_by_id[sample["sample_id"]]
        category = sample["category"]
        if category not in self.category_to_id:
            raise KeyError(f"Category {category!r} missing from category_to_id.")
        if bool(cached.get("is_unseen_category", False)):
            raise ValueError(
                f"Unseen category sample cannot provide centroid sketch: category={category}, sample_id={sample['sample_id']}"
            )
        local_z = int(cached["local_z"])
        num_modes = int(cached["num_modes_for_category"])
        if num_modes <= 0 or num_modes > self.max_num_modes_per_category:
            raise ValueError(
                f"Invalid num_modes_for_category for sample_id={sample['sample_id']}: {num_modes} not in [1, {self.max_num_modes_per_category}]"
            )
        if local_z < 0 or local_z >= num_modes:
            raise ValueError(f"Invalid local_z for sample_id={sample['sample_id']}: local_z={local_z} num_modes={num_modes}")
        fallback_reason = None
        prior = None
        if self.prior_source == "instruction_matrix_grammar":
            try:
                prior = self._instruction_grammar_prior(category, str(sample["sample_id"]))
            except InstructionGrammarPriorError as error:
                fallback_reason = "instruction_matrix_grammar_missing_or_invalid"
                if not self.fallback_if_missing:
                    raise ValueError(
                        f"{error} Set fallback_if_missing=true to fallback to category_kmeans, "
                        "or use --prior-source category_kmeans for old cache/checkpoint."
                    ) from error
                if self.fallback_prior_source != "category_kmeans":
                    raise ValueError(f"Unsupported fallback_prior_source={self.fallback_prior_source!r}; expected 'category_kmeans'.")
                prior = None
        if prior is None:
            prior = self._centroid_prior(category, local_z)
        centroid_fg_mask_prob = prior["fg_mask_prob"]
        centroid_fg_mask_bin = prior["fg_mask_bin"]
        centroid_label_prob_16 = prior["label_prob_16"]
        if not isinstance(centroid_fg_mask_prob, list) or len(centroid_fg_mask_prob) != 1 or not isinstance(centroid_fg_mask_prob[0], list) or len(centroid_fg_mask_prob[0]) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in centroid_fg_mask_prob[0]):
            raise ValueError(f"centroid_fg_mask_prob for sample_id={sample['sample_id']} must have shape [1,20,20].")
        if not isinstance(centroid_fg_mask_bin, list) or len(centroid_fg_mask_bin) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in centroid_fg_mask_bin):
            raise ValueError(f"centroid_fg_mask for sample_id={sample['sample_id']} must have shape [20,20].")
        if not isinstance(centroid_label_prob_16, list) or len(centroid_label_prob_16) != 16 or any(not isinstance(channel, list) or len(channel) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in channel) for channel in centroid_label_prob_16):
            raise ValueError(f"centroid_label_prob_16 for sample_id={sample['sample_id']} must have shape [16,20,20].")
        if sum(int(value) for row in centroid_fg_mask_bin for value in row) <= 0:
            raise ValueError(f"centroid_fg_mask for sample_id={sample['sample_id']} category={category!r} local_z={local_z} is all-zero.")
        prior_mode_id = int(prior["mode_id"])
        prior_source = str(prior["source"])
        prior_num_modes = int(prior.get("num_modes", num_modes)) if isinstance(prior.get("num_modes", num_modes), int | float) else num_modes
        condition_num_modes = max(1, min(self.max_num_modes_per_category, prior_num_modes if prior_source == "instruction_matrix_grammar" else num_modes))
        mode_mask = [1 if mode_index < condition_num_modes else 0 for mode_index in range(self.max_num_modes_per_category)]
        if prior_mode_id >= condition_num_modes:
            raise ValueError(f"Prior mode id out of range for sample_id={sample['sample_id']}: mode_id={prior_mode_id} condition_num_modes={condition_num_modes}")
        self._bump_prior_stats(category, prior_source, prior_mode_id, fallback_reason=fallback_reason)
        return {
            "sample_id": str(sample["sample_id"]),
            "category": str(category),
            "category_id": int(self.category_to_id[category]),
            "local_z": getattr(torch, "tensor")(prior_mode_id if prior_source == "instruction_matrix_grammar" else local_z, dtype=getattr(torch, "long")),
            "mode_mask": getattr(torch, "tensor")(mode_mask, dtype=getattr(torch, "bool")),
            "fg_y20": getattr(torch, "tensor")(cached["fg_y20"], dtype=getattr(torch, "long")),
            "fg_mask20": getattr(torch, "tensor")(cached["fg_mask20"], dtype=getattr(torch, "float32")),
            "bbox_stats": getattr(torch, "tensor")(cached["bbox_stats"], dtype=getattr(torch, "float32")),
            "label_hist_16": getattr(torch, "tensor")(cached["label_hist_16"], dtype=getattr(torch, "float32")),
            "row_projection": getattr(torch, "tensor")(cached["row_projection"], dtype=getattr(torch, "float32")),
            "col_projection": getattr(torch, "tensor")(cached["col_projection"], dtype=getattr(torch, "float32")),
            "grammar_signature": getattr(torch, "tensor")(cached["grammar_signature"], dtype=getattr(torch, "float32")),
            "adjacency_signature": getattr(torch, "tensor")(cached["adjacency_signature"], dtype=getattr(torch, "float32")),
            "descriptor": getattr(torch, "tensor")(cached["descriptor"], dtype=getattr(torch, "float32")),
            "fg_area": getattr(torch, "tensor")(float(cached["fg_area"]), dtype=getattr(torch, "float32")),
            "centroid_fg_mask_prob": getattr(torch, "tensor")(centroid_fg_mask_prob, dtype=getattr(torch, "float32")),
            "centroid_fg_mask_bin": getattr(torch, "tensor")(centroid_fg_mask_bin, dtype=getattr(torch, "float32")).unsqueeze(0),
            "centroid_label_prob_16": getattr(torch, "tensor")(centroid_label_prob_16, dtype=getattr(torch, "float32")),
            "centroid_label_hist": getattr(torch, "tensor")(prior.get("label_hist", [0.0] * 16), dtype=getattr(torch, "float32")),
            "centroid_row_projection": getattr(torch, "tensor")(prior.get("row_projection", [0.0] * 20), dtype=getattr(torch, "float32")),
            "centroid_col_projection": getattr(torch, "tensor")(prior.get("col_projection", [0.0] * 20), dtype=getattr(torch, "float32")),
            "centroid_adjacency": getattr(torch, "tensor")(prior.get("adjacency", [0.0] * 256), dtype=getattr(torch, "float32")),
            "centroid_transition_stats": getattr(torch, "tensor")(prior.get("transition_stats", [0.0] * 6), dtype=getattr(torch, "float32")),
            "centroid_bbox_stats": getattr(torch, "tensor")(prior.get("bbox_stats", [0.0] * 10), dtype=getattr(torch, "float32")),
            "prior_mode_id": getattr(torch, "tensor")(prior_mode_id, dtype=getattr(torch, "long")),
            "prior_source": prior_source,
            "original_y20": getattr(torch, "tensor")(cached["original_y20"], dtype=getattr(torch, "long")),
            "canonical_mode": str(self.canonical_mode),
            "metadata": cached,
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    torch, _ = _require_torch()
    return {
        "sample_ids": [sample["sample_id"] for sample in batch],
        "categories": [sample["category"] for sample in batch],
        "category_ids": getattr(torch, "tensor")([int(sample["category_id"]) for sample in batch], dtype=getattr(torch, "long")),
        "local_z": getattr(torch, "stack")([sample["local_z"] for sample in batch]),
        "mode_mask": getattr(torch, "stack")([sample["mode_mask"] for sample in batch]),
        "fg_y20": getattr(torch, "stack")([sample["fg_y20"] for sample in batch]),
        "fg_mask20": getattr(torch, "stack")([sample["fg_mask20"] for sample in batch]),
        "bbox_stats": getattr(torch, "stack")([sample["bbox_stats"] for sample in batch]),
        "label_hist_16": getattr(torch, "stack")([sample["label_hist_16"] for sample in batch]),
        "row_projection": getattr(torch, "stack")([sample["row_projection"] for sample in batch]),
        "col_projection": getattr(torch, "stack")([sample["col_projection"] for sample in batch]),
        "grammar_signature": getattr(torch, "stack")([sample["grammar_signature"] for sample in batch]),
        "adjacency_signature": getattr(torch, "stack")([sample["adjacency_signature"] for sample in batch]),
        "descriptor": getattr(torch, "stack")([sample["descriptor"] for sample in batch]),
        "fg_area": getattr(torch, "stack")([sample["fg_area"] for sample in batch]),
        "centroid_fg_mask_prob": getattr(torch, "stack")([sample["centroid_fg_mask_prob"] for sample in batch]),
        "centroid_fg_mask_bin": getattr(torch, "stack")([sample["centroid_fg_mask_bin"] for sample in batch]),
        "centroid_label_prob_16": getattr(torch, "stack")([sample["centroid_label_prob_16"] for sample in batch]),
        "centroid_label_hist": getattr(torch, "stack")([sample["centroid_label_hist"] for sample in batch]),
        "centroid_row_projection": getattr(torch, "stack")([sample["centroid_row_projection"] for sample in batch]),
        "centroid_col_projection": getattr(torch, "stack")([sample["centroid_col_projection"] for sample in batch]),
        "centroid_adjacency": getattr(torch, "stack")([sample["centroid_adjacency"] for sample in batch]),
        "centroid_transition_stats": getattr(torch, "stack")([sample["centroid_transition_stats"] for sample in batch]),
        "centroid_bbox_stats": getattr(torch, "stack")([sample["centroid_bbox_stats"] for sample in batch]),
        "prior_mode_id": getattr(torch, "stack")([sample["prior_mode_id"] for sample in batch]),
        "prior_sources": [sample["prior_source"] for sample in batch],
        "original_y20": getattr(torch, "stack")([sample["original_y20"] for sample in batch]),
        "canonical_mode": batch[0]["canonical_mode"] if batch else "full_masked",
        "metadata": [sample["metadata"] for sample in batch],
    }


def build_dataloader(
    manifest_path: str | Path,
    cache_path: str | Path,
    batch_size: int,
    shuffle: bool,
    category_to_id: dict[str, int] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    exclude_unseen_categories: bool = False,
    prior_source: str = "category_kmeans",
    prior_mode_sampling: str = "sample",
    prior_label_prob_key: str = "basis_label_prob_16",
    mode_prior_key: str = "mode_prior_smoothed",
    fallback_prior_source: str = "category_kmeans",
    fallback_if_missing: bool = True,
    prior_seed: int = 0,
) -> tuple[object, ForegroundDataset]:
    _, data = _require_torch()
    dataset = ForegroundDataset(
        manifest_path,
        cache_path=cache_path,
        category_to_id=category_to_id,
        exclude_unseen_categories=exclude_unseen_categories,
        prior_source=prior_source,
        prior_mode_sampling=prior_mode_sampling,
        prior_label_prob_key=prior_label_prob_key,
        mode_prior_key=mode_prior_key,
        fallback_prior_source=fallback_prior_source,
        fallback_if_missing=fallback_if_missing,
        prior_seed=prior_seed,
    )
    loader = getattr(data, "DataLoader")(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers if num_workers > 0 else False),
    )
    return loader, dataset
