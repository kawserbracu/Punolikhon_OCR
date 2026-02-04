"""
Enhanced Hybrid Recognition Training with Accuracy Improvements.

Improvements over train_hybrid_recognition_grapheme.py:
1. V2 model with Pre-LayerNorm, SE blocks, GELU
2. Focal CTC Loss for hard example focus
3. Cosine annealing with warm restarts LR scheduler
4. Stronger augmentation (morphological, perspective, cutout)
5. Gradient accumulation for larger effective batch
6. 6× English upsampling
7. Robust checkpoint resume (model, optimizer, scheduler, epoch)
8. Mixed precision training (AMP) for RTX 4070 Super

Usage:
    python training/train_hybrid_v2_enhanced.py ^
        --data_dir "stupid_again/manifests_combined" ^
        --vocab_path "stupid_again/manifests_combined/vocab_grapheme.json" ^
        --output_name rec_v2_enhanced ^
        --batch_size 16 ^
        --epochs 50

Resume:
    python training/train_hybrid_v2_enhanced.py ^
        --data_dir "stupid_again/manifests_combined" ^
        --vocab_path "stupid_again/manifests_combined/vocab_grapheme.json" ^
        --output_name rec_v2_enhanced ^
        --resume auto
"""
import argparse
import json
import random
from pathlib import Path
import sys
import time
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_recognition import collate_fn
from training.trainer_base import ensure_dir
from hybrid_2_me import HybridHTR_STN_Transformer, BengaliGraphemeTokenizer

try:
    from hybrid_2_me import HybridHTR_STN_Transformer_V2  # type: ignore
except Exception:
    HybridHTR_STN_Transformer_V2 = None

try:
    from hybrid_2_me import FocalCTCLoss  # type: ignore
except Exception:
    FocalCTCLoss = None


class RecDatasetEnhanced(Dataset):
    """
    Enhanced recognition dataset with stronger augmentation.
    
    New augmentations:
    - Morphological erosion/dilation (variable stroke width)
    - Perspective transform (camera angle simulation)
    - CoarseDropout (occlusion robustness)
    - ColorJitter (paper/ink variation)
    """
    
    def __init__(
        self,
        rec_json: Path,
        vocab_path: Path,
        augment: bool = True,
        strong_augment: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        data = json.loads(Path(rec_json).read_text(encoding='utf-8'))
        if isinstance(data, list):
            items = data
        else:
            items = data.get('items', [])
        # Drop invalid samples with empty ground-truth text
        items = [it for it in items if (it.get('word_text') or '').strip()]
        
        if max_samples is not None and max_samples > 0:
            random.Random(seed).shuffle(items)
            items = items[:max_samples]
        
        self.items = items
        self.tok = BengaliGraphemeTokenizer()
        if vocab_path.exists():
            self.tok.load_vocab(vocab_path)
        
        if augment:
            aug_list = [
                # Geometric transforms
                A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=8,
                    p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Perspective(scale=(0.03, 0.08), p=0.3),  # NEW: camera angle
                A.ElasticTransform(alpha=2.0, sigma=30, p=0.25),
                A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.25),
                
                # Photometric transforms
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.3),  # NEW
                A.GaussNoise(std_range=(0.02, 0.1), p=0.3),  # Updated API
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.MotionBlur(blur_limit=5, p=0.15),
                A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 1.0), p=0.2),
            ]
            
            if strong_augment:
                aug_list.extend([
                    # Morphological transforms (stroke width variation)
                    A.Morphological(scale=(2, 3), operation='erosion', p=0.15),  # NEW: thinner
                    A.Morphological(scale=(2, 3), operation='dilation', p=0.15),  # NEW: thicker
                    # Occlusion simulation - updated API for newer albumentations
                    A.CoarseDropout(
                        num_holes_range=(1, 6), hole_height_range=(4, 12), hole_width_range=(4, 12),
                        fill=255, p=0.2
                    ),  # NEW
                ])
            
            aug_list.append(ToTensorV2())
            self.transform = A.Compose(aug_list)
        else:
            self.transform = A.Compose([ToTensorV2()])
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        it = self.items[idx]
        img = cv2.imread(it['crop_path'])
        if img is None:
            img = np.zeros((128, 512, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize height to 128
        h, w = img.shape[:2]
        scale = 128.0 / max(1, h)
        new_w = max(int(round(w * scale)), 1)
        img = cv2.resize(img, (new_w, 128), interpolation=cv2.INTER_CUBIC)
        
        out = self.transform(image=img)
        img_t = out['image']
        
        if img_t.dtype != torch.float32:
            img_t = img_t.float()
        if torch.max(img_t) > 1.0:
            img_t = img_t / 255.0
        
        text = it.get('word_text') or ''
        target = torch.tensor(self.tok.encode_word(text), dtype=torch.long)
        
        return img_t, target


def greedy_ctc_decode(logits: torch.Tensor, blank: int = 0) -> List[int]:
    """Greedy CTC decoding."""
    pred = logits.argmax(dim=-1).squeeze(1).tolist()
    collapsed: List[int] = []
    prev = None
    for p in pred:
        if p == blank:
            prev = p
            continue
        if p != prev:
            collapsed.append(p)
        prev = p
    return collapsed


def cer(a: str, b: str) -> float:
    """Character Error Rate."""
    if len(b) == 0:
        return 0.0 if len(a) == 0 else 1.0
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + (a[i - 1] != b[j - 1]),
            )
    return float(dp[n, m] / max(1, len(b)))


def split_targets(flat_targets: torch.Tensor, target_lengths: torch.Tensor) -> List[List[int]]:
    """Split flat targets by lengths."""
    targets: List[List[int]] = []
    idx = 0
    for ln in target_lengths.tolist():
        ln = int(ln)
        targets.append([int(x) for x in flat_targets[idx: idx + ln].tolist()])
        idx += ln
    return targets


def evaluate_greedy_metrics(
    model: nn.Module,
    loader: DataLoader,
    tok: BengaliGraphemeTokenizer,
    device: torch.device,
) -> tuple:
    """Evaluate CER and word accuracy with greedy decoding."""
    model.eval()
    total_cer = 0.0
    correct_words = 0
    total = 0
    
    with torch.no_grad():
        for images, flat_targets, _, target_lengths in loader:
            images = images.to(device)
            logits = model(images)
            targets = split_targets(flat_targets, target_lengths)
            
            for i in range(logits.shape[1]):
                pred_ids = greedy_ctc_decode(logits[:, i:i+1, :], blank=0)
                pred_text = tok.decode_indices(pred_ids)
                gt_text = tok.decode_indices(targets[i])
                total_cer += cer(pred_text, gt_text)
                if pred_text == gt_text:
                    correct_words += 1
                total += 1
    
    return float(total_cer / max(1, total)), float(correct_words / max(1, total))


def find_latest_checkpoint(save_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in the save directory."""
    checkpoints = list(save_dir.glob('checkpoint_epoch*.pt'))
    if not checkpoints:
        # Try checkpoint_best.pt
        best = save_dir / 'checkpoint_best.pt'
        if best.exists():
            return best
        return None
    
    # Sort by epoch number
    def get_epoch(p):
        try:
            return int(p.stem.replace('checkpoint_epoch', ''))
        except:
            return 0
    
    checkpoints.sort(key=get_epoch, reverse=True)
    return checkpoints[0]


def main():
    ap = argparse.ArgumentParser(description="Enhanced Hybrid V2 Recognition Training")
    
    # Data arguments
    ap.add_argument("--data_dir", type=str, required=True, help="Root containing recognition_train.json/val.json")
    ap.add_argument("--vocab_path", type=str, required=True, help="Path to vocab_grapheme.json")
    ap.add_argument("--output_name", type=str, required=True, help="Run name under data_dir/runs")
    
    # Model arguments
    ap.add_argument("--model_variant", type=str, default="v2", 
                    choices=["v1", "v2"], help="v1=original, v2=improved with SE+PreNorm")
    ap.add_argument("--backbone_weights", type=str, default="DEFAULT")
    ap.add_argument("--dropout", type=float, default=0.15, help="Dropout rate for V2 model")
    
    # Training arguments
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size (16 for RTX 4070 Super)")
    ap.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    ap.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    ap.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (0 to disable)")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--seed", type=int, default=42)
    
    # Loss arguments
    ap.add_argument("--loss", type=str, default="focal", choices=["ctc", "focal"],
                    help="Loss function: ctc or focal (focuses on hard samples)")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    ap.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha")
    
    # Scheduler arguments
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["plateau", "cosine"],
                    help="LR scheduler: plateau or cosine (with warm restarts)")
    ap.add_argument("--cosine_t0", type=int, default=10, help="Cosine annealing restart period")
    
    # Augmentation arguments
    ap.add_argument("--strong_augment", action="store_true", default=True,
                    help="Enable strong augmentation (morphological, cutout)")
    ap.add_argument("--no_strong_augment", action="store_false", dest="strong_augment")
    
    # English upsampling
    ap.add_argument("--english_upsample", type=float, default=6.0, 
                    help="Oversample English samples (6× recommended)")
    
    # Resume arguments
    ap.add_argument("--resume", type=str, default="", 
                    help="'auto' to find latest, or path to checkpoint/model weights")
    ap.add_argument("--resume_lr", type=float, default=0.0, 
                    help="Override LR when resuming (0 = keep from checkpoint)")
    
    # Mixed precision
    ap.add_argument("--amp", action="store_true", default=True, 
                    help="Use automatic mixed precision (recommended for RTX 4070)")
    ap.add_argument("--no_amp", action="store_false", dest="amp")
    
    # Sampling
    ap.add_argument("--max_train_samples", type=int, default=0, help="Limit training samples (0=all)")
    ap.add_argument("--max_val_samples", type=int, default=0, help="Limit validation samples (0=all)")
    
    args = ap.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup directories
    data_dir = Path(args.data_dir)
    out_dir = data_dir / "runs" / args.output_name
    ensure_dir(out_dir)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer
    tok = BengaliGraphemeTokenizer()
    tok.load_vocab(Path(args.vocab_path))
    print(f"Vocabulary size: {tok.vocab_size()}")
    
    # Create model
    use_v2 = (args.model_variant == "v2")
    if use_v2 and HybridHTR_STN_Transformer_V2 is None:
        print("WARNING: HybridHTR_STN_Transformer_V2 not available in hybrid_2_me; falling back to V1")
        use_v2 = False

    if use_v2:
        model = HybridHTR_STN_Transformer_V2(
            num_classes=tok.vocab_size(),
            backbone_weights=args.backbone_weights,
            dropout=args.dropout,
        )
        print("Using V2 model (Pre-LayerNorm, SE blocks, GELU)")
    else:
        model = HybridHTR_STN_Transformer(
            num_classes=tok.vocab_size(),
            backbone_weights=args.backbone_weights,
        )
        print("Using V1 model (original)")
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Setup scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.cosine_t0,
            T_mult=2,
            eta_min=args.min_lr,
        )
        print(f"Using CosineAnnealingWarmRestarts (T_0={args.cosine_t0}, T_mult=2)")
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=args.min_lr
        )
        print("Using ReduceLROnPlateau")
    
    # Setup loss
    blank_idx = tok.grapheme_to_idx.get(tok.BLANK, 0)
    use_focal = (args.loss == "focal")
    if use_focal and FocalCTCLoss is None:
        print("WARNING: FocalCTCLoss not available in hybrid_2_me; falling back to standard CTC loss")
        use_focal = False

    if use_focal:
        criterion = FocalCTCLoss(
            blank=blank_idx,
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
        )
        print(f"Using Focal CTC Loss (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    else:
        criterion = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
        print("Using standard CTC Loss")
    
    # Setup datasets
    train_json = data_dir / "recognition_train.json"
    val_json = data_dir / "recognition_val.json"
    
    max_train = args.max_train_samples if args.max_train_samples > 0 else None
    max_val = args.max_val_samples if args.max_val_samples > 0 else None
    
    train_ds = RecDatasetEnhanced(
        train_json,
        vocab_path=Path(args.vocab_path),
        augment=True,
        strong_augment=args.strong_augment,
        max_samples=max_train,
        seed=args.seed,
    )
    
    val_ds = RecDatasetEnhanced(
        val_json,
        vocab_path=Path(args.vocab_path),
        augment=False,
        strong_augment=False,
        max_samples=max_val,
        seed=args.seed,
    )
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Setup weighted sampler for English upsampling
    sampler = None
    if args.english_upsample > 1.0:
        weights = []
        english_count = 0
        for it in train_ds.items:
            lang = (it.get('language') or '').strip().lower()
            if lang == 'english':
                weights.append(float(args.english_upsample))
                english_count += 1
            else:
                weights.append(1.0)
        
        if english_count > 0:
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            print(f"English upsampling {args.english_upsample}× ({english_count} English samples)")
    
    # Setup dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
        drop_last=True,  # Better for gradient accumulation
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    
    # Mixed precision scaler
    if args.amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    # Metrics logging
    metrics_csv = out_dir / 'training_log_metrics.csv'
    if not metrics_csv.exists():
        metrics_csv.write_text(
            'epoch,train_loss,val_loss,lr,val_cer,val_word_accuracy,best_cer\n',
            encoding='utf-8'
        )
    
    # Training state
    start_epoch = 1
    best_val_cer = float('inf')
    best_word_acc = 0.0
    
    # Resume handling
    if args.resume:
        resume_path = None
        
        if args.resume == "auto":
            # Find latest checkpoint in output directory
            resume_path = find_latest_checkpoint(out_dir)
            if resume_path:
                print(f"Auto-resume: found {resume_path}")
            else:
                print("Auto-resume: no checkpoint found, starting fresh")
        elif Path(args.resume).exists():
            resume_path = Path(args.resume)
        
        if resume_path:
            print(f"Loading checkpoint from {resume_path}")
            state = torch.load(resume_path, map_location='cpu')
            
            if 'model_state' in state:
                # Full checkpoint
                model.load_state_dict(state['model_state'])
                optimizer.load_state_dict(state['optimizer_state'])
                if state.get('scheduler_state') is not None:
                    scheduler.load_state_dict(state['scheduler_state'])
                start_epoch = state.get('epoch', 0) + 1
                best_val_cer = state.get('metrics', {}).get('val_cer', float('inf'))
                best_word_acc = state.get('metrics', {}).get('val_word_acc', 0.0)
                print(f"Resumed from epoch {start_epoch - 1}, best CER={best_val_cer:.4f}")
            else:
                # Model weights only
                model.load_state_dict(state)
                print("Loaded model weights (no optimizer/scheduler state)")
            
            # Override LR if specified
            if args.resume_lr > 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = args.resume_lr
                print(f"Overriding learning rate to {args.resume_lr}")
    
    # Save config
    config = vars(args)
    config['start_epoch'] = start_epoch
    config['vocab_size'] = tok.vocab_size()
    config['train_samples'] = len(train_ds)
    config['val_samples'] = len(val_ds)
    (out_dir / 'config.json').write_text(
        json.dumps(config, indent=2, default=str),
        encoding='utf-8'
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch} to {args.epochs}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"{'='*60}\n")
    
    epoch_times = []
    
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (images, flat_targets, input_lengths, target_lengths) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            flat_targets = flat_targets.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward with AMP
            if args.amp:
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    T = log_probs.shape[0]
                    maxW = input_lengths.max().clamp(min=1)
                    scaled_ilens = torch.clamp(
                        (input_lengths.float() / maxW.float()) * float(T),
                        min=1.0, max=float(T)
                    ).round().long()
                    
                    if args.loss == "focal":
                        loss = criterion(log_probs, flat_targets, scaled_ilens, target_lengths)
                    else:
                        loss = criterion(log_probs, flat_targets, scaled_ilens, target_lengths)
                    
                    loss = loss / args.grad_accum
                
                scaler.scale(loss).backward()
            else:
                logits = model(images)
                log_probs = torch.log_softmax(logits, dim=-1)
                T = log_probs.shape[0]
                maxW = input_lengths.max().clamp(min=1)
                scaled_ilens = torch.clamp(
                    (input_lengths.float() / maxW.float()) * float(T),
                    min=1.0, max=float(T)
                ).round().long()
                
                if args.loss == "focal":
                    loss = criterion(log_probs, flat_targets, scaled_ilens, target_lengths)
                else:
                    loss = criterion(log_probs, flat_targets, scaled_ilens, target_lengths)
                
                loss = loss / args.grad_accum
                loss.backward()
            
            train_loss += loss.item() * args.grad_accum
            
            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                if args.amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                
                optimizer.zero_grad()
        
        train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, flat_targets, input_lengths, target_lengths in val_loader:
                images = images.to(device, non_blocking=True)
                flat_targets = flat_targets.to(device, non_blocking=True)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
                
                logits = model(images)
                log_probs = torch.log_softmax(logits, dim=-1)
                T = log_probs.shape[0]
                maxW = input_lengths.max().clamp(min=1)
                scaled_ilens = torch.clamp(
                    (input_lengths.float() / maxW.float()) * float(T),
                    min=1.0, max=float(T)
                ).round().long()
                
                # Use standard CTC for validation loss
                val_criterion = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
                loss = val_criterion(log_probs, flat_targets, scaled_ilens, target_lengths)
                val_loss += loss.item()
        
        val_loss = val_loss / max(1, len(val_loader))
        
        # Evaluate CER and word accuracy
        val_cer, val_word_acc = evaluate_greedy_metrics(model, val_loader, tok, device)
        
        # Step scheduler
        lr = optimizer.param_groups[0]['lr']
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_cer)
        
        # Save checkpoints
        is_best = val_cer < best_val_cer - 1e-6
        if is_best:
            best_val_cer = val_cer
            best_word_acc = val_word_acc
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'metrics': {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_cer': val_cer,
                'val_word_acc': val_word_acc,
                'best_val_cer': best_val_cer,
            },
        }
        
        # Always save latest
        torch.save(checkpoint, out_dir / f'checkpoint_epoch{epoch}.pt')
        
        # Save best model
        if is_best:
            torch.save(model.state_dict(), out_dir / 'model_best_cer.pt')
            torch.save(checkpoint, out_dir / 'checkpoint_best.pt')
        
        # Cleanup old checkpoints (keep last 3)
        checkpoints = sorted(out_dir.glob('checkpoint_epoch*.pt'), key=lambda p: int(p.stem.replace('checkpoint_epoch', '')))
        for old in checkpoints[:-3]:
            old.unlink()
        
        # Log metrics
        t1 = time.time()
        epoch_time = t1 - t0
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining = args.epochs - epoch
        eta_min = (avg_time * remaining) / 60
        
        # Write to CSV
        with metrics_csv.open('a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{lr:.2e},{val_cer:.6f},{val_word_acc:.6f},{best_val_cer:.6f}\n")
        
        # Print progress
        best_marker = " *BEST*" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"CER: {val_cer:.4f} | Acc: {val_word_acc*100:.2f}% | "
            f"LR: {lr:.2e} | Time: {epoch_time:.1f}s | ETA: {eta_min:.1f}m{best_marker}"
        )
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best CER: {best_val_cer:.4f}")
    print(f"Best Word Accuracy: {best_word_acc*100:.2f}%")
    print(f"Model saved to: {out_dir / 'model_best_cer.pt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
