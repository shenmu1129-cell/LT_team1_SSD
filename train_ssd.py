#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.datasets import build_dataset, collate_fn
from src.eval_utils import evaluate_detector
from src.ssd_model import build_model_from_config, load_model_weights


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Please install PyYAML: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train torchvision SSD on YOLO-format datasets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="outputs/ssd")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr-step-size", type=int, default=None)
    parser.add_argument("--lr-gamma", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--finetune-from", default=None)
    parser.add_argument("--eval-map-every", type=int, default=None)
    parser.add_argument("--quick-eval-samples", type=int, default=None)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device: torch.device, epoch: int) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    skipped_empty_batches = 0
    start = time.time()
    for step, (images, targets) in enumerate(loader, start=1):
        non_empty_indices = [
            idx for idx, target in enumerate(targets) if target["boxes"].shape[0] > 0
        ]
        if not non_empty_indices:
            skipped_empty_batches += 1
            if skipped_empty_batches <= 5:
                print(
                    f"epoch {epoch} step {step}: skipped batch with no valid target boxes",
                    flush=True,
                )
            continue

        if len(non_empty_indices) != len(targets):
            dropped_paths = [
                targets[idx].get("image_path", f"index={idx}")
                for idx in range(len(targets))
                if idx not in non_empty_indices
            ]
            print(
                f"epoch {epoch} step {step}: dropped {len(dropped_paths)} empty-target images: "
                + "; ".join(dropped_paths[:3]),
                flush=True,
            )

        images = [images[idx].to(device) for idx in non_empty_indices]
        kept_targets = [targets[idx] for idx in non_empty_indices]
        tensor_targets = [
            {key: value.to(device) for key, value in target.items() if torch.is_tensor(value)}
            for target in kept_targets
        ]
        loss_dict = model(images, tensor_targets)
        losses = sum(loss for loss in loss_dict.values())
        finite_losses = all(math.isfinite(float(loss.item())) for loss in loss_dict.values())
        if not finite_losses or not math.isfinite(float(losses.item())):
            debug_rows = []
            for target in kept_targets:
                boxes = target["boxes"]
                labels = target["labels"]
                debug_rows.append(
                    {
                        "image": target.get("image_path", ""),
                        "label": target.get("label_path", ""),
                        "num_boxes": int(boxes.shape[0]),
                        "labels": labels.tolist(),
                        "boxes_min": float(boxes.min().item()) if boxes.numel() else None,
                        "boxes_max": float(boxes.max().item()) if boxes.numel() else None,
                    }
                )
            raise RuntimeError(
                f"Non-finite loss at epoch={epoch}, step={step}: {loss_dict}. "
                f"Batch debug: {debug_rows}"
            )

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        running_loss += float(losses.item())
        num_batches += 1
        if step % 50 == 0:
            elapsed = time.time() - start
            loss_text = ", ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
            print(
                f"epoch {epoch} step {step}/{len(loader)} "
                f"loss={running_loss / num_batches:.4f} {loss_text} {elapsed:.1f}s",
                flush=True,
            )
    if num_batches == 0:
        raise RuntimeError("No valid training batches were found. Check label paths and class ids.")
    return running_loss / num_batches


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: Dict[str, float], cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def append_quick_eval(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    epochs = args.epochs or int(train_cfg.get("epochs", 80))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 16))
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 4))
    lr = args.lr if args.lr is not None else float(train_cfg.get("lr", 0.005))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(train_cfg.get("weight_decay", 0.0005))
    lr_step_size = args.lr_step_size if args.lr_step_size is not None else int(train_cfg.get("lr_step_size", 30))
    lr_gamma = args.lr_gamma if args.lr_gamma is not None else float(train_cfg.get("lr_gamma", 0.1))
    eval_map_every = args.eval_map_every if args.eval_map_every is not None else int(train_cfg.get("eval_map_every", 10))
    quick_eval_samples = (
        args.quick_eval_samples
        if args.quick_eval_samples is not None
        else int(train_cfg.get("quick_eval_samples", 0))
    )
    score_threshold = float(train_cfg.get("score_threshold", 0.3))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_dataset(cfg, "train", data_root=args.data_root, skip_empty=True)
    val_dataset = build_dataset(cfg, "val", data_root=args.data_root, skip_empty=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_config(cfg).to(device)
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    start_epoch = 1
    best_loss = float("inf")
    best_map50 = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        metrics = checkpoint.get("metrics", {})
        best_loss = float(metrics.get("best_loss", best_loss))
        best_map50 = float(metrics.get("best_map50", best_map50))
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)
    elif args.finetune_from:
        _, missing, unexpected = load_model_weights(model, args.finetune_from, device)
        print(
            f"Loaded model weights from {args.finetune_from}; "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        metrics = {
            "epoch_loss": epoch_loss,
            "best_loss": min(best_loss, epoch_loss),
            "best_map50": best_map50,
            "lr": optimizer.param_groups[0]["lr"],
        }
        save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, epoch, metrics, cfg)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            metrics["best_loss"] = best_loss
            save_checkpoint(output_dir / "best_loss.pth", model, optimizer, scheduler, epoch, metrics, cfg)

        if quick_eval_samples and quick_eval_samples > 0:
            quick_metrics = evaluate_detector(
                model,
                val_dataset,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                score_threshold=score_threshold,
                max_samples=quick_eval_samples,
            )
            append_quick_eval(
                output_dir / "quick_eval.csv",
                {
                    "epoch": epoch,
                    "loss": f"{epoch_loss:.6f}",
                    "map50": f"{quick_metrics['map50']:.6f}",
                    "recall": f"{quick_metrics['recall']:.6f}",
                    "tp": int(quick_metrics["tp"]),
                    "gt": int(quick_metrics["gt"]),
                    "pred": int(quick_metrics["pred"]),
                },
            )
            print(
                f"quick eval epoch {epoch}: "
                f"mAP50={quick_metrics['map50']:.4f} recall={quick_metrics['recall']:.4f}",
                flush=True,
            )

        if eval_map_every and epoch % eval_map_every == 0:
            full_metrics = evaluate_detector(
                model,
                val_dataset,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                score_threshold=score_threshold,
            )
            print(
                f"full eval epoch {epoch}: "
                f"mAP50={full_metrics['map50']:.4f} recall={full_metrics['recall']:.4f}",
                flush=True,
            )
            if full_metrics["map50"] > best_map50:
                best_map50 = full_metrics["map50"]
                metrics["best_map50"] = best_map50
                metrics.update({f"val_{k}": v for k, v in full_metrics.items()})
                save_checkpoint(output_dir / "best_map50.pth", model, optimizer, scheduler, epoch, metrics, cfg)

    print(f"Training done. Checkpoints saved to {output_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    main()
