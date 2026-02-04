import argparse
import json
import random
from pathlib import Path
import sys
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_recognition import collate_fn, RecTrainer  # reuse trainer + collate
from training.trainer_base import ensure_dir
from hybrid_2_me import HybridHTR_STN_Transformer, HybridHTR_STN_Transformer_BiLSTM
from hybrid_2_me.tokenizer_grapheme import BengaliGraphemeTokenizer


class RecDatasetGrapheme(Dataset):
    def __init__(self, rec_json: Path, vocab_path: Path, augment=True, max_samples: int | None = None, seed: int = 42):
        data = json.loads(Path(rec_json).read_text(encoding='utf-8'))
        items = data.get('items', [])
        if max_samples is not None and max_samples > 0:
            random.Random(seed).shuffle(items)
            items = items[:max_samples]
        self.items = items
        self.tok = BengaliGraphemeTokenizer()
        if vocab_path.exists():
            self.tok.load_vocab(vocab_path)
        self.transform = A.Compose([
            A.Rotate(limit=7, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=7, p=0.4, border_mode=cv2.BORDER_CONSTANT),
            A.ElasticTransform(alpha=1.5, sigma=25, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.15),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.25),
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
    targets: List[List[int]] = []
    idx = 0
    for ln in target_lengths.tolist():
        ln = int(ln)
        targets.append([int(x) for x in flat_targets[idx: idx + ln].tolist()])
        idx += ln
    return targets


def evaluate_greedy_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    tok: BengaliGraphemeTokenizer,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_cer = 0.0
    correct_words = 0
    total = 0
    with torch.no_grad():
        for images, flat_targets, _, target_lengths in loader:
            images = images.to(device)
            logits = model(images)  # (T,N,V)
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


def main():
    ap = argparse.ArgumentParser(description="Train hybrid STN+Transformer recognition with grapheme tokenizer")
    ap.add_argument("--data_dir", type=str, required=True, help="Root containing recognition_train.json/val.json")
    ap.add_argument("--vocab_path", type=str, required=True)
    ap.add_argument("--output_name", type=str, required=True, help="Run name under data_dir/runs")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--backbone_weights", type=str, default="DEFAULT")
    ap.add_argument("--model_variant", type=str, default="hybrid_2_me", choices=["hybrid_2_me", "hybrid_2_me_bilstm"])
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint or model_best.pt to resume")
    ap.add_argument("--resume_lr", type=float, default=0.0, help="Override LR when resuming (0 = keep original)")
    ap.add_argument("--resume_epochs", type=int, default=0, help="Run this many more epochs after resume (0 = use max_epochs)")
    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze CNN backbone during training")
    ap.add_argument("--no_weight_decay", action="store_true", help="Disable weight decay for fine-tuning")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "runs" / args.output_name
    ensure_dir(out_dir)

    tok = BengaliGraphemeTokenizer()
    tok.load_vocab(Path(args.vocab_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.model_variant == "hybrid_2_me_bilstm":
        model = HybridHTR_STN_Transformer_BiLSTM(num_classes=tok.vocab_size(), backbone_weights=args.backbone_weights)
    else:
        model = HybridHTR_STN_Transformer(num_classes=tok.vocab_size(), backbone_weights=args.backbone_weights)
    model.to(device)

    # Optionally freeze backbone
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if 'backbone' in name or 'stn' in name:
                param.requires_grad = False
        print("Frozen backbone and STN layers")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()

    wd = 0.0 if args.no_weight_decay else 1e-4
    optimizer = AdamW(trainable_params, lr=2e-4, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

    train_json = data_dir / "recognition_train.json"
    val_json = data_dir / "recognition_val.json"

    max_train = args.max_train_samples if args.max_train_samples > 0 else None
    max_val = args.max_val_samples if args.max_val_samples > 0 else None

    train_ds = RecDatasetGrapheme(train_json, vocab_path=Path(args.vocab_path), augment=True, max_samples=max_train, seed=args.seed)
    val_ds = RecDatasetGrapheme(val_json, vocab_path=Path(args.vocab_path), augment=False, max_samples=max_val, seed=args.seed)

    nw = max(0, int(args.num_workers))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=False,
    )

    trainer = RecTrainer(
        model,
        optimizer,
        scheduler,
        patience=10,
        save_dir=out_dir,
        grad_clip_norm=5.0,
        label_smoothing=0.1,
        keep_last_n_checkpoints=2,
    )

    metrics_csv = out_dir / 'training_log_metrics.csv'
    if not metrics_csv.exists():
        metrics_csv.write_text('epoch,train_loss,val_loss,lr,val_cer,val_word_accuracy\n', encoding='utf-8')
    best_val_cer = float('inf')
    bad_epochs = 0
    patience_cer = 5
    min_epochs = 30
    enable_cer_early_stop = False

    start_epoch = 1
    end_epoch = args.max_epochs
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            # Check if this is a full checkpoint or just model weights (model_best.pt)
            state = torch.load(ckpt_path, map_location='cpu')
            if 'model_state' in state:
                # Full checkpoint with optimizer state
                last_epoch = trainer.load_checkpoint(ckpt_path)
                start_epoch = max(1, last_epoch + 1)
                print(f"Resumed from full checkpoint {ckpt_path} at epoch {last_epoch}")
            else:
                # model_best.pt style: just model weights, no optimizer state
                model.load_state_dict(state)
                start_epoch = 1  # Treat as epoch 1 since we don't know the original epoch
                print(f"Loaded model weights from {ckpt_path} (no optimizer state)")
            
            # Override LR if specified
            if args.resume_lr > 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = args.resume_lr
                print(f"Overriding learning rate to {args.resume_lr}")
            
            # Override end_epoch if resume_epochs specified
            if args.resume_epochs > 0:
                end_epoch = start_epoch + args.resume_epochs - 1
                print(f"Will train for {args.resume_epochs} more epochs (epochs {start_epoch} to {end_epoch})")
            else:
                print(f"Continuing from epoch {start_epoch} to {args.max_epochs}")

    for epoch in range(start_epoch, end_epoch + 1):
        tr_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate_epoch(val_loader)
        trainer.step_scheduler(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        val_cer, val_word_acc = evaluate_greedy_metrics(model, val_loader, tok, device)
        trainer.log_epoch(epoch, float(tr_loss), float(val_loss), float(lr), 0.0)
        with metrics_csv.open('a', encoding='utf-8') as f:
            f.write(f"{epoch},{tr_loss},{val_loss},{lr},{val_cer},{val_word_acc}\n")

        if val_cer < best_val_cer - 1e-6:
            best_val_cer = val_cer
            bad_epochs = 0
            trainer.save_checkpoint(epoch, {"val_loss": float(val_loss), "val_cer": float(val_cer)})
            torch.save(model.state_dict(), out_dir / 'model_best_cer.pt')
        else:
            bad_epochs += 1
            trainer.save_checkpoint(epoch, {"val_loss": float(val_loss), "val_cer": float(val_cer)})

        if enable_cer_early_stop and epoch >= min_epochs and bad_epochs >= patience_cer:
            break

    trainer.plot_curves()
    print("Finished hybrid grapheme recognition training. Logs at", out_dir)


if __name__ == "__main__":
    main()
