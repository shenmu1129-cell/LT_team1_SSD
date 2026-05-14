from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou

from .datasets import collate_fn


@torch.inference_mode()
def evaluate_detector(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    score_threshold: float = 0.3,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    if max_samples is not None:
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model.eval()
    predictions: List[Dict[str, torch.Tensor]] = []
    targets_all: List[Dict[str, torch.Tensor]] = []
    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            keep = output["scores"].detach().cpu() >= score_threshold
            predictions.append(
                {
                    "boxes": output["boxes"].detach().cpu()[keep],
                    "labels": output["labels"].detach().cpu()[keep],
                    "scores": output["scores"].detach().cpu()[keep],
                }
            )
            targets_all.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
            )

    return compute_map50_recall(predictions, targets_all)


def compute_map50_recall(
    predictions: Sequence[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    labels = sorted(
        {
            int(label)
            for target in targets
            for label in target["labels"].tolist()
            if int(label) > 0
        }
    )
    if not labels:
        return {"map50": 0.0, "recall": 0.0, "tp": 0.0, "gt": 0.0, "pred": 0.0}

    ap_values: List[float] = []
    total_tp = 0
    total_gt = 0
    total_pred = 0
    for class_label in labels:
        ap, tp, gt, pred = _average_precision_for_class(
            predictions, targets, class_label, iou_threshold
        )
        ap_values.append(ap)
        total_tp += tp
        total_gt += gt
        total_pred += pred

    recall = float(total_tp / total_gt) if total_gt else 0.0
    return {
        "map50": float(sum(ap_values) / len(ap_values)) if ap_values else 0.0,
        "recall": recall,
        "tp": float(total_tp),
        "gt": float(total_gt),
        "pred": float(total_pred),
    }


def _average_precision_for_class(
    predictions: Sequence[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
    class_label: int,
    iou_threshold: float,
) -> Tuple[float, int, int, int]:
    gt_by_image: Dict[int, Dict[str, torch.Tensor]] = {}
    total_gt = 0
    pred_rows: List[Tuple[float, int, torch.Tensor]] = []

    for image_idx, target in enumerate(targets):
        gt_keep = target["labels"] == class_label
        gt_boxes = target["boxes"][gt_keep]
        gt_by_image[image_idx] = {
            "boxes": gt_boxes,
            "matched": torch.zeros((gt_boxes.shape[0],), dtype=torch.bool),
        }
        total_gt += int(gt_boxes.shape[0])

    for image_idx, prediction in enumerate(predictions):
        pred_keep = prediction["labels"] == class_label
        for box, score in zip(prediction["boxes"][pred_keep], prediction["scores"][pred_keep]):
            pred_rows.append((float(score.item()), image_idx, box))

    pred_rows.sort(key=lambda row: row[0], reverse=True)
    if total_gt == 0:
        return 0.0, 0, 0, len(pred_rows)
    if not pred_rows:
        return 0.0, 0, total_gt, 0

    tp = torch.zeros((len(pred_rows),), dtype=torch.float32)
    fp = torch.zeros((len(pred_rows),), dtype=torch.float32)
    matched_tp = 0
    for pred_idx, (_, image_idx, pred_box) in enumerate(pred_rows):
        gt_info = gt_by_image[image_idx]
        gt_boxes = gt_info["boxes"]
        if gt_boxes.numel() == 0:
            fp[pred_idx] = 1.0
            continue
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_gt_idx = ious.max(dim=0)
        if best_iou >= iou_threshold and not bool(gt_info["matched"][best_gt_idx]):
            tp[pred_idx] = 1.0
            gt_info["matched"][best_gt_idx] = True
            matched_tp += 1
        else:
            fp[pred_idx] = 1.0

    precision = torch.cumsum(tp, dim=0) / torch.clamp(
        torch.cumsum(tp + fp, dim=0), min=1e-12
    )
    recall = torch.cumsum(tp, dim=0) / max(float(total_gt), 1e-12)
    ap = _voc_ap(recall, precision)
    return ap, matched_tp, total_gt, len(pred_rows)


def _voc_ap(recall: torch.Tensor, precision: torch.Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for idx in range(mpre.numel() - 1, 0, -1):
        mpre[idx - 1] = torch.maximum(mpre[idx - 1], mpre[idx])
    changing_points = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])
    return float(ap.item())


def write_metrics_csv(path: str | Path, rows: Iterable[Dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
