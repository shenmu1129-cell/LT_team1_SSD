from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collate_fn(batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _with_image_extensions(path: Path) -> Iterable[Path]:
    if path.suffix:
        yield path
    else:
        for ext in IMAGE_EXTENSIONS:
            yield path.with_suffix(ext)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _normalise_split(split: str, cfg: Dict[str, Any]) -> str:
    if split in ("val", "valid", "validation"):
        return cfg.get("val_split", "val")
    if split == "test":
        return cfg.get("test_split", "test")
    return cfg.get("train_split", "train")


def yolo_xywh_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height
    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    return x1, y1, x2, y2


class YOLODetectionDataset(Dataset):
    """YOLO-txt detection dataset adapted for torchvision detection models.

    Labels are converted from normalized YOLO ``class xc yc w h`` to pixel
    ``xyxy`` boxes. Class ids are shifted by +1 because torchvision SSD uses
    label 0 as background.
    """

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        split: str,
        class_names: Sequence[str],
        dataset_cfg: Optional[Dict[str, Any]] = None,
        skip_empty: bool = False,
        max_samples: Optional[int] = None,
        image_override_root: Optional[str | os.PathLike[str]] = None,
        image_suffix: str = "",
    ) -> None:
        self.data_root = Path(data_root).expanduser()
        self.split = split
        self.dataset_cfg = dataset_cfg or {}
        self.class_names = list(class_names)
        self.num_foreground_classes = len(self.class_names)
        self.skip_empty = skip_empty
        self.image_override_root = (
            Path(image_override_root).expanduser() if image_override_root else None
        )
        self.image_suffix = image_suffix
        self._override_index = self._build_override_index(self.image_override_root)

        base_samples = self._discover_base_samples()
        samples: List[Tuple[Path, Path, Path]] = []
        for base_image_path in base_samples:
            label_path = self._label_path_for(base_image_path)
            if skip_empty and not self._label_has_boxes(label_path):
                continue
            image_path = self._override_image_path(base_image_path)
            samples.append((image_path, label_path, base_image_path))
            if max_samples is not None and len(samples) >= max_samples:
                break

        if not samples:
            raise FileNotFoundError(
                f"No images found for split={split!r} under {self.data_root}"
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_path, label_path, base_image_path = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            image_tensor = F.to_tensor(image)

        boxes, labels = self._load_yolo_labels(label_path, width, height)
        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                if boxes.numel()
                else torch.zeros((0,), dtype=torch.float32)
            ),
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "base_image_path": str(base_image_path),
        }
        return image_tensor, target

    def _discover_base_samples(self) -> List[Path]:
        split_name = _normalise_split(self.split, self.dataset_cfg)
        list_key = {
            "train": "train_list",
            "val": "val_list",
            "valid": "val_list",
            "validation": "val_list",
            "test": "test_list",
        }.get(self.split, f"{self.split}_list")
        list_value = self.dataset_cfg.get(list_key)
        list_candidates: List[Path] = []
        if list_value:
            list_candidates.append(self.data_root / str(list_value))
        list_candidates.append(self.data_root / f"{split_name}.txt")

        list_path = _first_existing(list_candidates)
        if list_path:
            images = [
                self._resolve_list_image(line, split_name)
                for line in _read_lines(list_path)
            ]
            return [path for path in images if path is not None]

        image_dirs_cfg = self.dataset_cfg.get("image_dirs", {})
        configured_dir = image_dirs_cfg.get(self.split) or image_dirs_cfg.get(split_name)
        candidate_dirs = []
        if configured_dir:
            candidate_dirs.append(self.data_root / configured_dir)
        candidate_dirs.extend(
            [
                self.data_root / "images" / split_name,
                self.data_root / split_name / "images",
                self.data_root / split_name,
            ]
        )

        image_dir = _first_existing(candidate_dirs)
        if not image_dir:
            return []
        return sorted(path for path in image_dir.rglob("*") if _is_image(path))

    def _resolve_list_image(self, line: str, split_name: str) -> Optional[Path]:
        raw = Path(line)
        candidates: List[Path] = []
        if raw.is_absolute():
            candidates.extend(_with_image_extensions(raw))
        else:
            candidates.extend(_with_image_extensions(self.data_root / raw))
            candidates.extend(_with_image_extensions(self.data_root / split_name / "images" / raw.name))
            candidates.extend(_with_image_extensions(self.data_root / "images" / split_name / raw.name))
        return _first_existing(candidates)

    def _label_path_for(self, image_path: Path) -> Path:
        split_name = _normalise_split(self.split, self.dataset_cfg)
        label_dirs_cfg = self.dataset_cfg.get("label_dirs", {})
        configured_dir = label_dirs_cfg.get(self.split) or label_dirs_cfg.get(split_name)
        if configured_dir:
            candidate = self.data_root / configured_dir / f"{image_path.stem}.txt"
            if candidate.exists():
                return candidate

        parts = list(image_path.parts)
        candidates: List[Path] = []
        for idx, part in enumerate(parts):
            if part == "images":
                replaced = parts.copy()
                replaced[idx] = "labels"
                candidates.append(Path(*replaced).with_suffix(".txt"))
        candidates.extend(
            [
                self.data_root / split_name / "labels" / f"{image_path.stem}.txt",
                self.data_root / "labels" / split_name / f"{image_path.stem}.txt",
                image_path.with_suffix(".txt"),
            ]
        )
        return _first_existing(candidates) or candidates[0]

    def _load_yolo_labels(
        self, label_path: Path, image_width: int, image_height: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes: List[List[float]] = []
        labels: List[int] = []
        if not label_path.exists():
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        for line_no, line in enumerate(_read_lines(label_path), start=1):
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                class_id = int(float(fields[0]))
                x_center, y_center, width, height = map(float, fields[1:5])
            except ValueError as exc:
                raise ValueError(f"Bad YOLO label at {label_path}:{line_no}: {line}") from exc
            if class_id < 0 or class_id >= self.num_foreground_classes:
                continue
            x1, y1, x2, y2 = yolo_xywh_to_xyxy(
                x_center, y_center, width, height, image_width, image_height
            )
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id + 1)

        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    @staticmethod
    def _label_has_boxes(label_path: Path) -> bool:
        return label_path.exists() and any(line.split() for line in _read_lines(label_path))

    @staticmethod
    def _build_override_index(root: Optional[Path]) -> Dict[str, Path]:
        if root is None or not root.exists():
            return {}
        return {path.name: path for path in root.rglob("*") if _is_image(path)}

    def _override_image_path(self, base_image_path: Path) -> Path:
        if self.image_override_root is None:
            return base_image_path
        filename = f"{base_image_path.stem}{self.image_suffix}{base_image_path.suffix}"
        if filename in self._override_index:
            return self._override_index[filename]
        for ext in IMAGE_EXTENSIONS:
            alt_name = f"{base_image_path.stem}{self.image_suffix}{ext}"
            if alt_name in self._override_index:
                return self._override_index[alt_name]
        direct = self.image_override_root / filename
        if direct.exists():
            return direct
        raise FileNotFoundError(
            f"Could not find adversarial image for {base_image_path.name} in "
            f"{self.image_override_root} with suffix {self.image_suffix!r}"
        )


def build_dataset(
    cfg: Dict[str, Any],
    split: str,
    data_root: Optional[str] = None,
    skip_empty: Optional[bool] = None,
    max_samples: Optional[int] = None,
    image_override_root: Optional[str] = None,
    image_suffix: str = "",
) -> YOLODetectionDataset:
    dataset_cfg = cfg["dataset"]
    root = data_root or dataset_cfg["data_root"]
    class_names = dataset_cfg["class_names"]
    if skip_empty is None:
        skip_empty = bool(split == "train" and dataset_cfg.get("skip_empty_train", True))
    return YOLODetectionDataset(
        data_root=root,
        split=split,
        class_names=class_names,
        dataset_cfg=dataset_cfg,
        skip_empty=skip_empty,
        max_samples=max_samples,
        image_override_root=image_override_root,
        image_suffix=image_suffix,
    )
