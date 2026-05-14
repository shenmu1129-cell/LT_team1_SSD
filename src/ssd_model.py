from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torchvision.models.detection import ssd300_vgg16


def build_ssd_model(
    num_classes: int,
    pretrained_coco: bool = False,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
):
    """Build torchvision SSD300-VGG16.

    ``num_classes`` must include the background class. TT100K uses 46 and
    CCTSDB uses 4.
    """
    kwargs: Dict[str, Any] = {"num_classes": num_classes}
    if trainable_backbone_layers is not None:
        kwargs["trainable_backbone_layers"] = trainable_backbone_layers

    weights = None
    weights_backbone = None
    if pretrained_coco:
        try:
            from torchvision.models.detection import SSD300_VGG16_Weights

            weights = SSD300_VGG16_Weights.DEFAULT
            kwargs.pop("num_classes", None)
        except Exception:
            weights = "DEFAULT"
            kwargs.pop("num_classes", None)
    elif pretrained_backbone:
        try:
            from torchvision.models import VGG16_Weights

            weights_backbone = VGG16_Weights.IMAGENET1K_FEATURES
        except Exception:
            weights_backbone = "DEFAULT"

    model = ssd300_vgg16(
        weights=weights,
        weights_backbone=weights_backbone,
        **kwargs,
    )
    if pretrained_coco and num_classes != 91:
        model = _replace_ssd_classification_head(model, num_classes)
    return model


def _replace_ssd_classification_head(model, num_classes: int):
    from torchvision.models.detection.ssd import SSDClassificationHead

    old_head = model.head.classification_head
    old_num_classes = getattr(old_head, "num_classes", 91)
    in_channels = []
    num_anchors = []
    for layer in old_head.module_list:
        in_channels.append(layer.in_channels)
        num_anchors.append(layer.out_channels // old_num_classes)
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    return model


def build_model_from_config(cfg: Dict[str, Any]):
    class_names = cfg["dataset"]["class_names"]
    model_cfg = cfg.get("model", {})
    num_classes = len(class_names) + 1
    return build_ssd_model(
        num_classes=num_classes,
        pretrained_coco=bool(model_cfg.get("pretrained_coco", False)),
        pretrained_backbone=bool(model_cfg.get("pretrained_backbone", True)),
        trainable_backbone_layers=model_cfg.get("trainable_backbone_layers"),
    )


def load_model_weights(model, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return checkpoint, missing, unexpected
