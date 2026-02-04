import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_recognition import RecDataset, collate_fn, RecTrainer  # reuse dataset/ctc trainer
from data.tokenizer import BengaliWordOCRTokenizer
from hybrid_model import HybridHTR_STN_Transformer
from training.trainer_base import ensure_dir


def main():
    ap = argparse.ArgumentParser(description="Train hybrid STN+Transformer recognition model")
    ap.add_argument("--data_dir", type=str, required=True, help="Root containing recognition_train.json/val.json")
    ap.add_argument("--vocab_path", type=str, required=True)
    ap.add_argument("--output_name", type=str, required=True, help="Run name under data_dir/runs")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0, help="Set 0 on Windows to avoid hang")
    ap.add_argument("--backbone_weights", type=str, default="DEFAULT", help="Torchvision weights for MobileNetV3")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "runs" / args.output_name
    ensure_dir(out_dir)

    tok = BengaliWordOCRTokenizer()
    tok.load_vocab(Path(args.vocab_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = HybridHTR_STN_Transformer(num_classes=tok.vocab_size(), backbone_weights=args.backbone_weights)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

    train_json = data_dir / "recognition_train.json"
    val_json = data_dir / "recognition_val.json"
    train_ds = RecDataset(train_json, vocab_path=Path(args.vocab_path), augment=True)
    val_ds = RecDataset(val_json, vocab_path=Path(args.vocab_path), augment=False)

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
        patience=25,
        save_dir=out_dir,
        grad_clip_norm=5.0,
        label_smoothing=0.1,
        keep_last_n_checkpoints=3,
    )

    max_epochs = 200
    min_epochs = 30
    for epoch in range(1, max_epochs + 1):
        tr_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate_epoch(val_loader)
        trainer.step_scheduler(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        trainer.log_epoch(epoch, float(tr_loss), float(val_loss), float(lr), 0.0)
        trainer.save_checkpoint(epoch, {"val_loss": float(val_loss)})
        if epoch >= min_epochs and trainer.should_stop(val_loss):
            break

    trainer.plot_curves()
    print("Finished hybrid recognition training. Logs at", out_dir)


if __name__ == "__main__":
    main()
