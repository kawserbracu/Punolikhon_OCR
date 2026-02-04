import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detection_mnv3seg import MNV3Seg


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class DetMaskDataset(Dataset):
    """
    Builds binary masks from DocTR-style detection JSON entries.
    - Expects each item to have keys: img_path, img_dimensions (H, W), boxes (normalized [x1,y1,x2,y2]).
    - On the fly: loads image, applies augs, builds target mask (shrunken word regions like DB).
    """
    def __init__(self, det_json: Path, augment: bool, project_root: Path, shrink_ratio: float = 0.8):
        data = json.loads(det_json.read_text(encoding='utf-8'))
        self.items: List[Dict[str, Any]] = data.get('items', [])
        self.project_root = project_root
        self.shrink_ratio = shrink_ratio
        if augment:
            self.tf = A.Compose([
                A.Rotate(limit=3, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.LongestMaxSize(max_size=1024, p=1.0),
                A.PadIfNeeded(min_height=768, min_width=1024, p=1.0),
                A.Resize(height=768, width=1024, p=1.0),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=1024, p=1.0),
                A.PadIfNeeded(min_height=768, min_width=1024, p=1.0),
                A.Resize(height=768, width=1024, p=1.0),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.items)

    def _resolve_img(self, path_hint: str) -> np.ndarray | None:
        img = cv2.imread(path_hint)
        if img is not None:
            return img
        name = Path(path_hint).name
        try:
            for p in self.project_root.rglob(name):
                img = cv2.imread(str(p))
                if img is not None:
                    return img
        except Exception:
            pass
        return None

    @staticmethod
    def _shrink_box(x1: float, y1: float, x2: float, y2: float, ratio: float) -> Tuple[int, int, int, int]:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        hw = 0.5 * (x2 - x1) * ratio
        hh = 0.5 * (y2 - y1) * ratio
        nx1 = int(round(cx - hw)); ny1 = int(round(cy - hh))
        nx2 = int(round(cx + hw)); ny2 = int(round(cy + hh))
        return nx1, ny1, nx2, ny2

    def __getitem__(self, idx):
        it = self.items[idx]
        img = self._resolve_img(it['img_path'])
        if img is None:
            img = np.zeros((768, 768, 3), dtype=np.uint8)
        H0, W0 = map(int, it.get('img_dimensions', (img.shape[0], img.shape[1])))
        # Build mask in original size then transform with same tf
        mask = np.zeros((H0, W0), dtype=np.uint8)
        for b in it.get('boxes', []):
            x1 = int(round(b[0] * W0)); y1 = int(round(b[1] * H0))
            x2 = int(round(b[2] * W0)); y2 = int(round(b[3] * H0))
            sx1, sy1, sx2, sy2 = self._shrink_box(x1, y1, x2, y2, self.shrink_ratio)
            sx1 = max(0, min(W0 - 1, sx1)); sx2 = max(0, min(W0 - 1, sx2))
            sy1 = max(0, min(H0 - 1, sy1)); sy2 = max(0, min(H0 - 1, sy2))
            if sx2 > sx1 and sy2 > sy1:
                mask[sy1:sy2, sx1:sx2] = 255
        # Apply transforms jointly
        out = self.tf(image=img, mask=mask)
        img_t = out['image']
        msk_t = out['mask']
        # Ensure float32 in [0,1]
        if img_t.dtype != torch.float32:
            img_t = img_t.float()
        if torch.max(img_t) > 1.0:
            img_t = img_t / 255.0
        msk_t = (msk_t.float() / 255.0).unsqueeze(0)  # (1,H,W)
        return img_t, msk_t


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice = 1 - ((2 * intersection + smooth) / (union + smooth))
        return bce + dice.mean()


def train_epoch(model, loader, optimizer, device, criterion) -> float:
    model.train()
    total = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        # Resize masks to logits size if needed
        if logits.shape[-2:] != masks.shape[-2:]:
            masks = torch.nn.functional.interpolate(masks, size=logits.shape[-2:], mode='nearest')
        loss = criterion(logits, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


def val_epoch(model, loader, device, criterion) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            if logits.shape[-2:] != masks.shape[-2:]:
                masks = torch.nn.functional.interpolate(masks, size=logits.shape[-2:], mode='nearest')
            loss = criterion(logits, masks)
            total += float(loss.item())
    return total / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser(description='Custom DB-like detector trainer (MobileNetV3 backbone + mask supervision)')
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--backbone', type=str, default='large', choices=['large','small','hybrid'])
    ap.add_argument('--output_name', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / 'runs' / args.output_name
    ensure_dir(out_dir)

    train_json = data_dir / 'detection_train.json'
    val_json = data_dir / 'detection_val.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = DetMaskDataset(train_json, augment=True, project_root=PROJECT_ROOT)
    val_ds = DetMaskDataset(val_json, augment=False, project_root=PROJECT_ROOT)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MNV3Seg(backbone=args.backbone, pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    criterion = BCEDiceLoss()

    best_val = float('inf')
    no_improve = 0
    max_epochs = 100
    patience = 15

    log_path = out_dir / 'training_log.csv'
    if not log_path.exists():
        log_path.write_text('epoch,train_loss,val_loss,lr\n', encoding='utf-8')

    for epoch in range(1, max_epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, criterion)
        vl = val_epoch(model, val_loader, device, criterion)
        scheduler.step(vl)
        lr = optimizer.param_groups[0]['lr']
        with log_path.open('a', encoding='utf-8') as f:
            f.write(f'{epoch},{tr:.6f},{vl:.6f},{lr}\n')

        if vl < best_val - 1e-5:
            best_val = vl
            no_improve = 0
            torch.save({'state_dict': model.state_dict(), 'backbone': args.backbone}, out_dir / 'model_best.pt')
        else:
            no_improve += 1

        if epoch % 5 == 0:
            torch.save({'state_dict': model.state_dict(), 'backbone': args.backbone}, out_dir / f'epoch_{epoch}.pt')

        if no_improve >= patience:
            break

    torch.save({'state_dict': model.state_dict(), 'backbone': args.backbone}, out_dir / 'model_last.pt')


if __name__ == '__main__':
    main()
