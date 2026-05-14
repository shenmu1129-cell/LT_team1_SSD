#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

from src.ssd_model import build_model_from_config, load_model_weights


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Please install PyYAML: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image SSD inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    class_names = cfg["dataset"]["class_names"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(device)
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    image_path = Path(args.image)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    with torch.inference_mode():
        output = model([image_tensor])[0]

    keep = output["scores"].detach().cpu() >= args.score_threshold
    boxes = output["boxes"].detach().cpu()[keep]
    labels = output["labels"].detach().cpu()[keep]
    scores = output["scores"].detach().cpu()[keep]

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for box, label_tensor, score_tensor in zip(boxes, labels, scores):
        label_id = int(label_tensor.item())
        score = float(score_tensor.item())
        class_name = class_names[label_id - 1] if 1 <= label_id <= len(class_names) else "background"
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        print(
            f"bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
            f"label_id={label_id} class={class_name} score={score:.4f}"
        )
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        caption = f"{label_id}:{class_name} {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        draw.rectangle(text_bbox, fill=(255, 0, 0))
        draw.text((x1, y1), caption, fill=(255, 255, 255), font=font)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(output_path)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
