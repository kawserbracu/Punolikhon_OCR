from __future__ import annotations
from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_frcnn_model(num_classes: int, pretrained: bool = True) -> Any:
    """
    Build a Faster R-CNN ResNet50-FPN detector.
    num_classes includes background as class 0. For word detection with a single class, use num_classes=2.
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    # replace the classifier with a new one, that has num_classes which is user-defined
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
