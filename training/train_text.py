"""
NSFW Content Filter — Text Model Training Script

Training pipeline for the hybrid CNN + Bi-LSTM text classifier:
    - Vocabulary building from training data
    - Data loading with padding
    - Adam optimizer with LR scheduling
    - Early stopping based on validation F1
    - Checkpoint + vocabulary saving
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.text_model import TextCNN_BiLSTM, Vocabulary

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping: filename-based convention
# Files: safe_texts.txt (one text per line), nsfw_texts.txt (one text per line)


# ===========================================================================
# Dataset
# ===========================================================================

class NSFWTextDataset(Dataset):
    """
    Text dataset loaded from plain text files.

    Expects:
        data_dir/safe_texts.txt  → label 0 (one text per line)
        data_dir/nsfw_texts.txt  → label 1 (one text per line)
    """

    def __init__(
        self,
        data_dir: str,
        vocab: Vocabulary,
        max_seq_len: int = 256,
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[str, int]] = []

        data_path = Path(data_dir)

        # Load safe texts
        safe_file = data_path / "safe_texts.txt"
        if safe_file.exists():
            with open(safe_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append((line, 0))

        # Load NSFW texts
        nsfw_file = data_path / "nsfw_texts.txt"
        if nsfw_file.exists():
            with open(nsfw_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append((line, 1))

        logger.info(
            "Loaded %d texts from %s (safe: %d, nsfw: %d)",
            len(self.samples), data_dir,
            sum(1 for _, l in self.samples if l == 0),
            sum(1 for _, l in self.samples if l == 1),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoded = self.vocab.encode(text, max_length=self.max_seq_len)
        return torch.tensor(encoded, dtype=torch.long), label

    def get_all_texts(self) -> List[str]:
        """Return all raw texts for vocabulary building."""
        return [text for text, _ in self.samples]


# ===========================================================================
# Trainer
# ===========================================================================

class TextTrainer:
    """Manages the full training lifecycle for the text model."""

    def __init__(
        self,
        data_dir: str = "data_text",
        checkpoint_dir: str = "checkpoints",
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        patience: int = 7,
        max_seq_len: int = 256,
        max_vocab_size: int = 30000,
        embed_dim: int = 128,
    ):
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.max_seq_len = max_seq_len

        # --- Build vocabulary from training data ---
        logger.info("Building vocabulary from training data...")
        temp_texts = self._load_raw_texts(str(self.data_dir / "train"))
        self.vocab = Vocabulary(max_vocab_size=max_vocab_size, min_freq=2)
        self.vocab.build(temp_texts)
        logger.info("Vocabulary size: %d", len(self.vocab))

        # --- Datasets ---
        self.train_dataset = NSFWTextDataset(
            str(self.data_dir / "train"), self.vocab, max_seq_len
        )
        self.val_dataset = NSFWTextDataset(
            str(self.data_dir / "val"), self.vocab, max_seq_len
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        # --- Model ---
        self.model = TextCNN_BiLSTM(
            vocab_size=len(self.vocab),
            embed_dim=embed_dim,
            num_classes=2,
            max_seq_len=max_seq_len,
        ).to(DEVICE)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model created on %s — %d params", DEVICE, total_params)

        # --- Loss, optimizer, scheduler ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

        # --- Tracking ---
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.history = []

    @staticmethod
    def _load_raw_texts(data_dir: str) -> List[str]:
        """Load all raw texts from a directory for vocab building."""
        texts = []
        path = Path(data_dir)
        for filename in ("safe_texts.txt", "nsfw_texts.txt"):
            fpath = path / filename
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    texts.extend(line.strip() for line in f if line.strip())
        return texts

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in self.train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(self.train_loader), 1)
        return {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
            "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in self.val_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(self.val_loader), 1)
        return {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
            "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
        }

    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint and vocabulary."""
        # Save model
        model_path = self.checkpoint_dir / "best_text_model.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "vocab_size": len(self.vocab),
        }, model_path)

        # Save vocabulary
        vocab_path = self.checkpoint_dir / "vocabulary.json"
        self.vocab.save(str(vocab_path))

        logger.info("Checkpoint saved: %s (F1: %.4f)", model_path, metrics["f1"])

    def train(self):
        """Run the full training loop."""
        logger.info("=" * 60)
        logger.info("Starting text model training — %d epochs", self.epochs)
        logger.info("=" * 60)

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step(val_metrics["f1"])

            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d/%d (%.1fs) — LR: %.6f | "
                "Train Loss: %.4f, F1: %.4f | "
                "Val Loss: %.4f, F1: %.4f",
                epoch, self.epochs, elapsed, lr,
                train_metrics["loss"], train_metrics["f1"],
                val_metrics["loss"], val_metrics["f1"],
            )

            self.history.append({
                "epoch": epoch,
                "lr": lr,
                "train": train_metrics,
                "val": val_metrics,
            })

            # Save best model
            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Save history
        hist_path = self.checkpoint_dir / "text_training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info("Training complete. Best Val F1: %.4f", self.best_f1)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train text NSFW classifier")
    parser.add_argument("--data-dir", type=str, default="data_text")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=30000)
    args = parser.parse_args()

    trainer = TextTrainer(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        max_seq_len=args.max_seq_len,
        max_vocab_size=args.vocab_size,
    )
    trainer.train()


if __name__ == "__main__":
    main()
