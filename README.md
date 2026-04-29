# Knit

Minimal Python pipeline for turning stitch-code images into QA artifacts.

## Setup

```bash
python -m pip install -e .
```

For PyTorch Dataset/DataLoader helpers:

```bash
python -m pip install -e .[train]
```

## Run

Process the default `Tuck` and `Hem` categories from `dataset/` into `outputs/`:

```bash
python -m knit_decode
```

Useful options:

```bash
python -m knit_decode --limit 10 --output-dir outputs/dev-run
python -m knit_decode --categories Tuck Hem --cell-width 8 --cell-height 8
python -m knit_decode --categories Tuck Hem --export-ar --output-dir outputs/ar-run
```

Each sample is written to `outputs/<category>/<source_file_stem>/` with:

- `cropped_source.png`
- `quantized.png`
- `grid_overlay.png`
- `decoded_grid.tsv`
- `decoded_grid.json`
- `reconstructed.png`
- `diff.png`
- `metrics.json`

When `--export-ar` is enabled, each sample also gets:

- `ar_id_grid.json`
- `ar_token_sequence.json`
- `ar_token_ids.txt`

`metrics.json` includes the inferred cell size plus any right/bottom pixels trimmed because the cropped image was not a perfect multiple of the inferred grid.

Run-level summaries are written to:

- `outputs/run_summary.json`
- `outputs/samples.csv`

When `--export-ar` is enabled, the run root also gets:

- `outputs/ar_manifest.jsonl`
- `outputs/ar_vocab.json`

## PyTorch loading

Use the generated AR exports with the dataset/dataloader helpers:

```python
from knit_decode.ar_dataset import ArExportDataset, build_ar_dataloader

dataset = ArExportDataset("outputs/ar-run")
print(len(dataset), dataset[0]["sample_id"])

dataloader = build_ar_dataloader("outputs/ar-run", batch_size=2)
batch = next(iter(dataloader))
print(batch["input_ids"].shape, batch["grid_ids"].shape)
```

Or run the example script:

```bash
python examples/ar_dataloader.py outputs/ar-run --batch-size 2
```

## PixelCNN baseline

Run the PixelCNN control baseline on exported knit grids:

```bash
python -m knit_decode.pixelcnn_baseline \
  --export-root outputs/ar-run \
  --output-dir results/pixelcnn-smoke \
  --train --evaluate --generate \
  --batch-size 2 --n-epochs 1
```

This baseline reuses `external/pixel_models/pixelcnn.py`, but feeds it local `ar_id_grid.json` exports through a thin adapter in `knit_decode/pixelcnn_dataset.py`.

## Notes

- Legend colors are loaded from `dataset/all_info.json`.
- Duplicate legend colors are preserved and surfaced as candidate action ids instead of being silently collapsed.
- Simulation images are discovered when a matching normalized stem exists under `dataset/simulation images/`.
