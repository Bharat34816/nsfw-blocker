"""
NSFW Content Filter — Image Model Training Script

Complete training pipeline for the custom EfficientNet-B0:
    - Data loading with augmentation
    - AdamW optimizer with cosine annealing LR
    - Mixed-precision training
    - Gradient clipping & early stopping
    - Checkpoint saving (best F1 score)
    - Epoch-level metric logging
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.efficientnet_model import EfficientNetB0

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Dataset
# ===========================================================================

class NSFWImageDataset(Dataset):
    """
    Dataset that loads images from a directory structure:
        root/safe/   → label 0
        root/nsfw/   → label 1
    """

    LABEL_MAP = {"safe": 0, "nsfw": 1}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for label_name, label_idx in self.LABEL_MAP.items():
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                continue
            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    self.samples.append((str(img_path), label_idx))

        logger.info(
            "Loaded %d images from %s (%s)",
            len(self.samples),
            root_dir,
            {v: sum(1 for _, l in self.samples if l == v) for v in self.LABEL_MAP.values()},
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


# ===========================================================================
# Data Augmentation
# ===========================================================================

def get_train_transforms() -> transforms.Compose:
    """Training transforms with aggressive augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])


def get_val_transforms() -> transforms.Compose:
    """Validation / inference transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ===========================================================================
# Training Loop
# ===========================================================================

class ImageTrainer:
    """
    Manages the full training lifecycle for EfficientNet-B0.
    """

    def __init__(
        self,
        data_dir: str = "data_processed",
        checkpoint_dir: str = "checkpoints",
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        patience: int = 10,
        num_workers: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.num_workers = num_workers

        # Model
        self.model = EfficientNetB0(num_classes=2).to(DEVICE)
        logger.info("Model created on %s", DEVICE)

        # Loss, Optimizer, Scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        self.scaler = GradScaler()

        # Dataloaders
        self.train_loader = self._make_loader("train", get_train_transforms())
        self.val_loader = self._make_loader("val", get_val_transforms())

        # Tracking
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.history = []

    def _make_loader(self, split: str, transform) -> DataLoader:
        """Create a DataLoader for a specific split."""
        dataset = NSFWImageDataset(
            root_dir=str(self.data_dir / split),
            transform=transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Returns metrics dict."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed-precision forward pass
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

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
        """Evaluate on validation set. Returns metrics dict."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in self.val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast():
                outputs = self.model(images)
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
        """Save model checkpoint."""
        path = self.checkpoint_dir / "best_image_model.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }, path)
        logger.info("Checkpoint saved: %s (F1: %.4f)", path, metrics["f1"])

    def train(self):
        """Run the full training loop with early stopping."""
        logger.info("=" * 60)
        logger.info("Starting training — %d epochs, patience=%d", self.epochs, self.patience)
        logger.info("=" * 60)

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()

            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d/%d (%.1fs) — LR: %.6f | "
                "Train Loss: %.4f, Acc: %.4f, F1: %.4f | "
                "Val Loss: %.4f, Acc: %.4f, F1: %.4f",
                epoch, self.epochs, elapsed, lr,
                train_metrics["loss"], train_metrics["accuracy"], train_metrics["f1"],
                val_metrics["loss"], val_metrics["accuracy"], val_metrics["f1"],
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
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training complete. Best Val F1: %.4f", self.best_f1)


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 for NSFW detection")
    parser.add_argument("--data-dir", type=str, default="data_processed", help="Path to processed data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    trainer = ImageTrainer(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.workers,
    )
    trainer.train()


if __name__ == "__main__":
    main()
