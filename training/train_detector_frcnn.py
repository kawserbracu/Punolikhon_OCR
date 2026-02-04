import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.detection_frcnn import build_frcnn_model
from training.trainer_base import ensure_dir


class WordDetDataset(Dataset):
    def __init__(self, det_json: Path, augment: bool, project_root: Path):
        data = json.loads(det_json.read_text(encoding='utf-8'))
        self.items: List[Dict[str, Any]] = data.get('items', [])
        self.project_root = project_root
        # Basic geometric + photometric augs; Albumentations with Pascal VOC boxes (absolute pixel coords)
        if augment:
            self.tf = A.Compose([
                A.Rotate(limit=5, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.LongestMaxSize(max_size=1024, p=1.0),
                A.PadIfNeeded(min_height=1024, min_width=1024, p=1.0),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=1024, p=1.0),
                A.PadIfNeeded(min_height=1024, min_width=1024, p=1.0),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.items)

    def _resolve_img(self, path_hint: str) -> np.ndarray | None:
        img = cv2.imread(path_hint)
        if img is not None:
            return img
        name = Path(path_hint).name
        # try project root rglob by name
        try:
            for p in self.project_root.rglob(name):
                img = cv2.imread(str(p))
                if img is not None:
                    return img
        except Exception:
            pass
        return None

    def __getitem__(self, idx):
        it = self.items[idx]
        img = self._resolve_img(it['img_path'])
        if img is None:
            img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        h_json, w_json = it['img_dimensions']
        H0, W0 = int(h_json), int(w_json)
        # boxes are normalized in JSON: [x1,y1,x2,y2]
        boxes_norm = np.array(it['boxes'], dtype=np.float32)
        # convert to absolute pixel coordinates for Pascal VOC
        boxes_abs = []
        for b in boxes_norm:
            x1 = float(b[0]) * W0
            y1 = float(b[1]) * H0
            x2 = float(b[2]) * W0
            y2 = float(b[3]) * H0
            boxes_abs.append([x1, y1, x2, y2])
        boxes_abs = np.array(boxes_abs, dtype=np.float32)
        labels = np.ones((len(boxes_abs),), dtype=np.int64)  # single class: word

        if self.tf:
            out = self.tf(image=img, bboxes=boxes_abs, labels=labels)
            img_t = out['image']
            # Ensure float32 in [0,1]
            if img_t.dtype != torch.float32:
                img_t = img_t.float()
            if torch.max(img_t) > 1.0:
                img_t = img_t / 255.0
            boxes_abs = np.array(out['bboxes'], dtype=np.float32)
            labels = out['labels']
        else:
            img_t = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1)).float() / 255.0

        target = {
            'boxes': torch.as_tensor(boxes_abs, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    losses_sum = 0.0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses_sum += float(loss.item())
    return losses_sum / max(1, len(loader))


def evaluate_loss(model, loader, device) -> float:
    model.train()  # FasterRCNN returns losses only in train mode
    losses_sum = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            losses_sum += float(loss.item())
    return losses_sum / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser(description='Train a word detector (Faster R-CNN) on DocTR-style detection JSONs')
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--output_name', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / 'runs' / args.output_name
    ensure_dir(out_dir)

    train_json = data_dir / 'detection_train.json'
    val_json = data_dir / 'detection_val.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = WordDetDataset(train_json, augment=True, project_root=PROJECT_ROOT)
    val_ds = WordDetDataset(val_json, augment=False, project_root=PROJECT_ROOT)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # One foreground class + background
    model = build_frcnn_model(num_classes=2, pretrained=True)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

    best_val = float('inf')
    no_improve = 0
    max_epochs = 100
    patience = 15

    # CSV log
    log_path = out_dir / 'training_log.csv'
    if not log_path.exists():
        log_path.write_text('epoch,train_loss,val_loss,lr\n', encoding='utf-8')

    for epoch in range(1, max_epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        vl = evaluate_loss(model, val_loader, device)
        scheduler.step(vl)
        lr = optimizer.param_groups[0]['lr']
        with log_path.open('a', encoding='utf-8') as f:
            f.write(f'{epoch},{tr:.6f},{vl:.6f},{lr}\n')

        # checkpoint best
        if vl < best_val - 1e-5:
            best_val = vl
            no_improve = 0
            torch.save(model.state_dict(), out_dir / 'model_best.pt')
        else:
            no_improve += 1

        # periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), out_dir / f'epoch_{epoch}.pt')

        if no_improve >= patience:
            break

    torch.save(model.state_dict(), out_dir / 'model_last.pt')


if __name__ == '__main__':
    main()
