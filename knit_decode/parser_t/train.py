from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .dataset import build_parser_dataloader
from .losses import shift_tolerant_cross_entropy
from .model import TinyTopologyParser


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser training. Install with `pip install -e .[train]`.") from error
    return torch, optim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a first-pass simulation-image to topology parser.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest mapping simulation images to stitch-code color-map targets.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=1, help="Global shift tolerance used by the Inverse-Knitting-style CE.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(128, 128), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch, optim = _require_torch()
    dataloader, dataset = build_parser_dataloader(
        args.manifest,
        batch_size=args.batch_size,
        shuffle=True,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
    )
    model = TinyTopologyParser(num_classes=dataset.palette.num_classes)
    optimizer = getattr(optim, "Adam")(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in dataloader:
            images = batch["images"]
            targets = batch["targets"]
            logits = model(images)
            loss = shift_tolerant_cross_entropy(logits, targets, max_shift=args.max_shift)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1

        mean_loss = total_loss / max(1, batch_count)
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"epoch={epoch + 1} loss={mean_loss:.6f}")

    metrics = {
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_shift": args.max_shift,
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "learning_rate": args.learning_rate,
        "num_classes": dataset.palette.num_classes,
        "num_samples": len(dataset),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")(
        {
            "model_state_dict": model.state_dict(),
            "palette_colors": [list(color) for color in dataset.palette.colors],
            "metrics": metrics,
        },
        args.output_dir / "checkpoint.pt",
    )
    print(f"saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"saved checkpoint: {args.output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
