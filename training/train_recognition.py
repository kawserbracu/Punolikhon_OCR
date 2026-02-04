import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.recognition import CRNN
from data.tokenizer import BengaliWordOCRTokenizer
from training.trainer_base import BaseTrainer, ensure_dir


class RecDataset(Dataset):
    def __init__(self, rec_json: Path, vocab_path: Path, augment=True):
        data = json.loads(Path(rec_json).read_text(encoding='utf-8'))
        self.items: List[Dict[str, Any]] = data.get('items', [])
        self.tok = BengaliWordOCRTokenizer()
        if vocab_path.exists():
            self.tok.load_vocab(vocab_path)
        self.augment = augment
        # Enhanced augmentation for handwritten data (REMOVED CoarseDropout - counterproductive)
        self.transform = A.Compose([
            A.Rotate(limit=7, p=0.5, border_mode=cv2.BORDER_CONSTANT),  # Increased rotation probability and range
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=7, p=0.4, border_mode=cv2.BORDER_CONSTANT),  # More aggressive
            A.ElasticTransform(alpha=1.5, sigma=25, p=0.3),  # Increased for more handwriting variations
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),  # Increased distortion
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),  # More brightness/contrast variation
            A.GaussNoise(p=0.3),  # Increased noise probability
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),  # Stronger blur
            A.MotionBlur(blur_limit=5, p=0.15),  # Increased motion blur
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.25),  # Increased sharpening
            ToTensorV2(),
        ]) if augment else A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = cv2.imread(it['crop_path'])
        if img is None:
            img = np.zeros((128, 512, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize height to 128 (increased from 32 for better feature extraction)
        h, w = img.shape[:2]
        scale = 128.0 / max(1, h)
        new_w = max(int(round(w * scale)), 1)
        img = cv2.resize(img, (new_w, 128), interpolation=cv2.INTER_CUBIC)  # Use CUBIC for upscaling
        out = self.transform(image=img)
        img_t = out['image']  # (C,H,W)
        # Ensure float32 in [0,1]
        if img_t.dtype != torch.float32:
            img_t = img_t.float()
        if torch.max(img_t) > 1.0:
            img_t = img_t / 255.0
        text = it.get('word_text') or ''
        target = torch.tensor(self.tok.encode_word(text), dtype=torch.long)
        return img_t, target


def collate_fn(batch):
    # Pad images to same width, concatenate targets, compute lengths
    imgs, targets = zip(*batch)
    C, H = imgs[0].shape[0], imgs[0].shape[1]
    widths = [im.shape[2] for im in imgs]
    maxW = max(widths)
    padded = []
    for im in imgs:
        # Ensure same height (should already be 32); if not, resize
        if im.shape[1] != H:
            im = torch.nn.functional.interpolate(im.unsqueeze(0), size=(H, im.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
        padW = maxW - im.shape[2]
        if padW > 0:
            pad = torch.zeros((C, H, padW), dtype=im.dtype)
            im = torch.cat([im, pad], dim=2)
        padded.append(im)
    images = torch.stack(padded, dim=0)
    # Targets
    flat_targets = torch.cat(targets) if len(targets) > 0 else torch.zeros(0, dtype=torch.long)
    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    # Input lengths from original widths before padding (CRNN averages height only)
    input_lengths = torch.tensor(widths, dtype=torch.long)
    return images, flat_targets, input_lengths, target_lengths


class RecTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, patience, save_dir, grad_clip_norm=5.0, label_smoothing=0.1, keep_last_n_checkpoints=3):
        super().__init__(model, optimizer, scheduler, patience, save_dir, keep_last_n_checkpoints=keep_last_n_checkpoints)
        self.ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.grad_clip_norm = grad_clip_norm
        self.label_smoothing = label_smoothing

    def _compute_train_loss(self, batch) -> torch.Tensor:
        images, flat_targets, input_lengths, target_lengths = batch
        dev = next(self.model.parameters()).device
        images = images.to(dev, non_blocking=True)
        flat_targets = flat_targets.to(dev, non_blocking=True)
        input_lengths = input_lengths.to(dev)
        target_lengths = target_lengths.to(dev)
        logits = self.model(images)  # (T, N, V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        T, N, V = log_probs.shape
        
        # Apply label smoothing to CTC targets
        if self.label_smoothing > 0 and self.model.training:
            # Smooth the log_probs distribution
            smooth_probs = log_probs * (1 - self.label_smoothing) + (torch.log(torch.tensor(1.0 / V, device=dev)) * self.label_smoothing)
            log_probs = smooth_probs
        
        # Scale input lengths from pixel widths to feature time-steps
        maxW = input_lengths.max().clamp(min=1)
        scaled_ilens = torch.clamp((input_lengths.float() / maxW.float()) * float(T), min=1.0, max=float(T)).round().long()
        # CTC expects (T, N, C)
        loss = self.ctc(log_probs, flat_targets, scaled_ilens, target_lengths)
        return loss
    
    def train_epoch(self, loader):
        """Override to add gradient clipping"""
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            self.optimizer.zero_grad()
            loss = self._compute_train_loss(batch)
            loss.backward()
            # Apply gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser(description='Train CRNN recognition')
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--vocab_path', type=str, required=True)
    ap.add_argument('--output_name', type=str, required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / 'runs' / args.output_name
    ensure_dir(out_dir)

    tok = BengaliWordOCRTokenizer()
    tok.load_vocab(Path(args.vocab_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    model = CRNN(vocab_size=tok.vocab_size(), pretrained_backbone=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=2e-4, weight_decay=1e-4, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)  # Increased patience for scheduler

    train_json = data_dir / 'recognition_train.json'
    val_json = data_dir / 'recognition_val.json'

    train_ds = RecDataset(train_json, vocab_path=Path(args.vocab_path), augment=True)
    val_ds = RecDataset(val_json, vocab_path=Path(args.vocab_path), augment=False)

    # Batch size 12 for larger combined dataset (reduced model size allows slightly larger batches)
    # Use worker processes + pinned memory to keep GPU fed.
    train_loader = DataLoader(
        train_ds, batch_size=12, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=12, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
    )

    # Patience 25, gradient clipping 5.0, label smoothing 0.1, keep only last 3 checkpoints
    trainer = RecTrainer(model, optimizer, scheduler, patience=25, save_dir=out_dir, 
                        grad_clip_norm=5.0, label_smoothing=0.1, keep_last_n_checkpoints=3)

    # Verify loss on first batch
    images, flat_targets, input_lengths, target_lengths = next(iter(train_loader))
    images = images.to(device)
    flat_targets = flat_targets.to(device)
    input_lengths = input_lengths.to(device)
    target_lengths = target_lengths.to(device)
    with torch.no_grad():
        logits = model(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        T = log_probs.shape[0]
        maxW = input_lengths.max().clamp(min=1)
        scaled_ilens = torch.clamp((input_lengths.float() / maxW.float()) * float(T), min=1.0, max=float(T)).round().long()
        _ = trainer.ctc(log_probs, flat_targets, scaled_ilens, target_lengths)

    max_epochs = 200  # Increased to 200 as requested
    min_epochs = 30
    for epoch in range(1, max_epochs + 1):
        tr_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate_epoch(val_loader)
        trainer.step_scheduler(val_loss)
        lr = optimizer.param_groups[0]['lr']
        trainer.log_epoch(epoch, float(tr_loss), float(val_loss), float(lr), 0.0)
        trainer.save_checkpoint(epoch, {'val_loss': float(val_loss)})
        if epoch >= min_epochs and trainer.should_stop(val_loss):
            break

    trainer.plot_curves()
    print('Finished recognition training. Logs at', out_dir)


if __name__ == '__main__':
    main()
