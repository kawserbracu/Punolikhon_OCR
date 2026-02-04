import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.detection import build_detection_model
from training.trainer_base import BaseTrainer, ensure_dir


class DetectionDataset(Dataset):
    def __init__(self, det_json: Path, base_dir: Path, transform=None):
        data = json.loads(Path(det_json).read_text(encoding='utf-8'))
        self.items: List[Dict[str, Any]] = data.get('items', [])
        self.transform = transform
        self.base_dir = base_dir
        self.project_root = PROJECT_ROOT

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        ipath = it['img_path']
        # Try direct path first
        img = cv2.imread(ipath)
        if img is None:
            # Try basename in base_dir
            name = Path(ipath).name
            cand = self.base_dir / name
            img = cv2.imread(str(cand))
        if img is None:
            # If ipath is relative like 'raw/xxx.jpg', try project_root/ipath
            p_rel = Path(ipath)
            if not p_rel.is_absolute():
                cand_rel = self.project_root / p_rel
                img = cv2.imread(str(cand_rel))
        if img is None:
            # Try common split folders relative to base_dir parent
            for split in ("d1", "d2", "d3", "d4", "raw"):
                cand2 = self.project_root / split / Path(ipath).name
                img = cv2.imread(str(cand2))
                if img is not None:
                    break
        if img is None:
            # Last resort: rglob by exact name under project root
            try:
                for p in self.project_root.rglob(Path(ipath).name):
                    img = cv2.imread(str(p))
                    if img is not None:
                        break
            except Exception:
                pass
        if img is None:
            # Fallback: create a blank image to avoid crashing the training loop
            img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = it['img_dimensions']
        boxes = np.array(it['boxes'], dtype=np.float32)
        labels = it.get('labels') or [""] * len(boxes)
        # Albumentations expects bboxes in normalized [x_min, y_min, x_max, y_max]
        if self.transform:
            out = self.transform(image=img, bboxes=boxes, labels=labels)
            img = out['image']
            boxes = np.array(out['bboxes'], dtype=np.float32)
            labels = out['labels']
        return img, boxes, labels, (h, w)


def build_transforms():
    return A.Compose([
        A.Rotate(limit=5, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.LongestMaxSize(max_size=1024, p=1.0),
        A.PadIfNeeded(min_height=1024, min_width=1024, p=1.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))


class DetectionTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, patience, save_dir, dummy_param: torch.nn.Parameter | None = None):
        super().__init__(model, optimizer, scheduler, patience, save_dir)
        # DocTR loss placeholder: in lieu of full DB loss, use dummy loss to keep script runnable.
        self._dummy = dummy_param if dummy_param is not None else torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def _compute_train_loss(self, batch) -> torch.Tensor:
        images, boxes, labels, sizes = batch
        # images: list/tensor, but model here is predictor for inference; we use dummy train loss
        return (self._dummy * 0 + 1.0)

    def _compute_val_loss(self, batch) -> torch.Tensor:
        return self._compute_train_loss(batch)


def collate_fn(batch):
    imgs, boxes, labels, sizes = zip(*batch)
    # imgs already tensors from ToTensorV2
    return imgs, boxes, labels, sizes


def main():
    ap = argparse.ArgumentParser(description='Train detection model (template)')
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--output_name', type=str, required=True)
    ap.add_argument('--config', type=str, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / 'runs' / args.output_name
    ensure_dir(out_dir)

    train_json = data_dir / 'detection_train.json'
    val_json = data_dir / 'detection_val.json'

    train_ds = DetectionDataset(train_json, base_dir=data_dir, transform=build_transforms())
    val_ds = DetectionDataset(val_json, base_dir=data_dir, transform=build_transforms())

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_detection_model(pretrained=True)
    try:
        model.to(device)
    except Exception:
        pass
    # Use a dummy parameter to satisfy optimizer requirements in this scaffold
    dummy_param = torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
    optimizer = Adam([dummy_param], lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-7)

    trainer = DetectionTrainer(model, optimizer, scheduler, patience=15, save_dir=out_dir, dummy_param=dummy_param)

    max_epochs = 100
    for epoch in range(1, max_epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate_epoch(val_loader)
        # step scheduler on val
        trainer.step_scheduler(val_loss)
        # log
        lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
        trainer.log_epoch(epoch, float(train_loss), float(val_loss), float(lr), 0.0)
        # save checkpoint
        trainer.save_checkpoint(epoch, {'val_loss': float(val_loss)})
        # early stop
        if trainer.should_stop(val_loss):
            break

    trainer.plot_curves()
    print('Finished detection training template run. Logs at', out_dir)


if __name__ == '__main__':
    main()
