import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class History:
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    lrs: List[float] = field(default_factory=list)


class BaseTrainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: Optional[ReduceLROnPlateau], patience: int, save_dir: str | Path,
                 keep_last_n_checkpoints: int = 3):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.save_dir = Path(save_dir)
        ensure_dir(self.save_dir)
        self.best_val = float('inf')
        self.bad_epochs = 0
        self.start_epoch = 1
        self.history = History()
        self.csv_path = self.save_dir / 'training_log.csv'
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.saved_checkpoints = []  # Track saved checkpoint files

        if not self.csv_path.exists():
            with self.csv_path.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'time_sec'])

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        start = time.time()
        for batch in train_loader:
            loss = self._compute_train_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
        elapsed = time.time() - start
        return total_loss / max(1, n)

    def validate_epoch(self, val_loader) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_val_loss(batch)
                total_loss += loss.item()
                n += 1
        return total_loss / max(1, n)

    def _compute_train_loss(self, batch) -> torch.Tensor:
        # Placeholder; override in scripts if needed
        raise NotImplementedError

    def _compute_val_loss(self, batch) -> torch.Tensor:
        return self._compute_train_loss(batch)

    def step_scheduler(self, val_loss: float) -> None:
        if self.scheduler is not None:
            self.scheduler.step(val_loss)

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_val - 1e-8:
            self.best_val = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> None:
        ckpt = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val': self.best_val,
            'bad_epochs': self.bad_epochs,
            'metrics': metrics,
        }
        
        # Save checkpoint for this epoch
        checkpoint_path = self.save_dir / f'checkpoint_epoch{epoch}.pt'
        torch.save(ckpt, checkpoint_path)
        self.saved_checkpoints.append(checkpoint_path)
        
        # Always update model_best.pt if this is the best model so far
        val_loss = metrics.get('val_loss', float('inf'))
        if val_loss <= self.best_val:
            torch.save(self.model.state_dict(), self.save_dir / 'model_best.pt')
            # Also save as best checkpoint
            torch.save(ckpt, self.save_dir / 'checkpoint_best.pt')
        
        # Clean up old checkpoints, keep only last N
        if len(self.saved_checkpoints) > self.keep_last_n_checkpoints:
            old_checkpoints = self.saved_checkpoints[:-self.keep_last_n_checkpoints]
            for old_path in old_checkpoints:
                if old_path.exists() and old_path.name != 'checkpoint_best.pt':
                    old_path.unlink()
            self.saved_checkpoints = self.saved_checkpoints[-self.keep_last_n_checkpoints:]

    def load_checkpoint(self, path: str | Path) -> int:
        ck = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ck['model_state'])
        self.optimizer.load_state_dict(ck['optimizer_state'])
        if self.scheduler is not None and ck.get('scheduler_state') is not None:
            self.scheduler.load_state_dict(ck['scheduler_state'])
        self.best_val = ck.get('best_val', float('inf'))
        self.bad_epochs = ck.get('bad_epochs', 0)
        return int(ck.get('epoch', 1))

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float, elapsed: float) -> None:
        self.history.epochs.append(epoch)
        self.history.train_loss.append(train_loss)
        self.history.val_loss.append(val_loss)
        self.history.lrs.append(lr)
        with self.csv_path.open('a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, val_loss, lr, round(elapsed, 2)])

    def plot_curves(self) -> None:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(self.history.epochs, self.history.train_loss, label='train_loss')
        ax1.plot(self.history.epochs, self.history.val_loss, label='val_loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.legend(loc='upper right')
        plt.tight_layout()
        fig.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close(fig)
