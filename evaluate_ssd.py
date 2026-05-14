#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from src.datasets import build_dataset
from src.eval_utils import evaluate_detector, write_metrics_csv
from src.ssd_model import build_model_from_config, load_model_weights


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Please install PyYAML: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SSD checkpoints on clean and adversarial images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--adv-root", default=None)
    parser.add_argument("--adv-suffix", default="")
    parser.add_argument("--source-model", default="clean")
    parser.add_argument("--target-detector", default="SSD")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-csv", default="outputs/ssd_eval_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(device)
    _, missing, unexpected = load_model_weights(model, args.checkpoint, device)
    if missing or unexpected:
        print(f"checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)}")

    clean_dataset = build_dataset(
        cfg,
        "test",
        data_root=args.data_root,
        skip_empty=False,
        max_samples=args.max_samples,
    )
    clean_metrics = evaluate_detector(
        model,
        clean_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        score_threshold=args.score_threshold,
    )

    adv_metrics = {"map50": 0.0, "recall": 0.0}
    asr = 0.0
    if args.adv_root:
        adv_dataset = build_dataset(
            cfg,
            "test",
            data_root=args.data_root,
            skip_empty=False,
            max_samples=args.max_samples,
            image_override_root=args.adv_root,
            image_suffix=args.adv_suffix,
        )
        adv_metrics = evaluate_detector(
            model,
            adv_dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            score_threshold=args.score_threshold,
        )
        if clean_metrics["recall"] > 0:
            asr = max(0.0, (clean_metrics["recall"] - adv_metrics["recall"]) / clean_metrics["recall"])

    row = {
        "Source Model": args.source_model,
        "Target Detector": args.target_detector,
        "Clean mAP50": f"{clean_metrics['map50']:.6f}",
        "Adv mAP50": f"{adv_metrics['map50']:.6f}" if args.adv_root else "",
        "Clean Recall": f"{clean_metrics['recall']:.6f}",
        "Adv Recall": f"{adv_metrics['recall']:.6f}" if args.adv_root else "",
        "ASR": f"{asr:.6f}" if args.adv_root else "",
    }
    headers = list(row.keys())
    widths = {key: max(len(key), len(str(value))) for key, value in row.items()}
    print(" | ".join(key.ljust(widths[key]) for key in headers))
    print(" | ".join("-" * widths[key] for key in headers))
    print(" | ".join(str(row[key]).ljust(widths[key]) for key in headers))

    write_metrics_csv(Path(args.output_csv), [row])
    print(f"Saved metrics CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
