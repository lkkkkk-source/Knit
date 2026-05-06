# Knit

Clean branch for the next research direction.

Current reset goal:

- keep the repository skeleton
- preserve datasets as local, ignored assets
- rebuild the training and generation pipeline around the new simulation-image objective

The previous topology-decoding and autoregressive baseline work has been archived on branch:

- `codex/archive-topology-pipeline`

## First parser-T attempt

The first rebuild target is:

- `simulation image -> topology parser T`

This branch now includes a minimal scaffold under `knit_decode/parser_t/`:

- `dataset.py`: manifest loading and parser dataset skeleton
- `losses.py`: Inverse-Knitting-style shift-tolerant cross-entropy
- `model.py`: lightweight dense prediction baseline
- `cli.py`: manifest builder entrypoint
- `train.py`: training CLI scaffold

### Install

```bash
python -m pip install -e .
python -m pip install -e .[train]
```

### Build a starter manifest

```bash
knit-parse-t build-manifest --dataset-root dataset_complete --output-path parser_t/manifest.jsonl
```

This only creates a starter manifest. You still need to generate topology target JSON files under `parser_t/targets/`.
