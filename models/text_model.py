"""
NSFW Content Filter — Text Classification Model (From Scratch)

A hybrid 1D-CNN + Bi-LSTM text classifier for detecting NSFW text.
No external pre-trained embeddings — vocabulary and embeddings are
learned from the training data.

Architecture:
    Embedding → [Conv1D × 3 (multi-scale)] → MaxPool → Concat
    → Bi-LSTM → Attention → FC → Sigmoid

This captures both local n-gram patterns (CNN) and long-range
sequential dependencies (LSTM).
"""

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Text Preprocessor & Vocabulary
# ===========================================================================

class Vocabulary:
    """
    Maps tokens to integer indices, built from training data.

    Special tokens:
        <PAD> = 0  (padding)
        <UNK> = 1  (unknown / out-of-vocabulary)
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_vocab_size: int = 30000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.token2idx: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.idx2token: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build(self, texts: List[str]) -> "Vocabulary":
        """Build vocabulary from a list of raw text strings."""
        counter: Counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Sort by frequency, take top tokens meeting min_freq
        most_common = [
            (tok, freq) for tok, freq in counter.most_common()
            if freq >= self.min_freq
        ]
        most_common = most_common[: self.max_vocab_size - 2]  # reserve PAD, UNK

        for tok, _ in most_common:
            idx = len(self.token2idx)
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

        return self

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower().strip()
        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s!?.,]", " ", text)
        # Split on whitespace
        tokens = text.split()
        return tokens

    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """Convert text to padded integer sequence."""
        tokens = self.tokenize(text)[:max_length]
        indices = [
            self.token2idx.get(tok, self.token2idx[self.UNK_TOKEN])
            for tok in tokens
        ]
        # Pad or truncate
        if len(indices) < max_length:
            indices += [self.token2idx[self.PAD_TOKEN]] * (max_length - len(indices))
        return indices

    def __len__(self) -> int:
        return len(self.token2idx)

    def save(self, path: str):
        """Save vocabulary to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from JSON."""
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.token2idx = json.load(f)
        vocab.idx2token = {v: k for k, v in vocab.token2idx.items()}
        return vocab


# ===========================================================================
# Attention Layer
# ===========================================================================

class Attention(nn.Module):
    """
    Simple additive attention over sequence outputs.

    Learns to weight each time-step based on importance,
    producing a single context vector.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(
        self, lstm_output: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
            mask: Optional (batch, seq_len) boolean mask for padding.

        Returns:
            context: (batch, hidden_size)
        """
        scores = self.attn(lstm_output).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=1)                  # (batch, seq_len)
        context = torch.bmm(
            weights.unsqueeze(1), lstm_output
        ).squeeze(1)   # (batch, hidden_size)

        return context


# ===========================================================================
# Text CNN + Bi-LSTM Classifier
# ===========================================================================

class TextCNN_BiLSTM(nn.Module):
    """
    Hybrid 1D-CNN + Bi-LSTM text classifier for NSFW detection.

    Pipeline:
        1. Embedding layer (learned from scratch)
        2. Multi-scale 1D convolutions (kernel sizes 3, 4, 5)
        3. Max-over-time pooling per kernel
        4. Concatenated CNN features fed into Bi-LSTM
        5. Attention-weighted aggregation
        6. Fully connected classification head

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension.
        num_filters: Number of filters per CNN kernel size.
        kernel_sizes: List of kernel sizes for multi-scale CNN.
        lstm_hidden: Hidden size for Bi-LSTM.
        lstm_layers: Number of LSTM layers.
        num_classes: Output classes (2 for Safe/NSFW).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: List[int] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        max_seq_len: int = 256,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_filters = num_filters

        # --- Embedding (learned from scratch) ---
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # Zero out padding embedding
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

        # --- Multi-scale 1D CNN ---
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, ks, padding=ks // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            )
            for ks in kernel_sizes
        ])

        cnn_out_dim = num_filters * len(kernel_sizes)

        # --- Bi-LSTM ---
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # --- Attention ---
        self.attention = Attention(lstm_out_dim)

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # --- Initialize weights ---
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming init for Conv/Linear, orthogonal for LSTM."""
        for name, param in self.named_parameters():
            if "embedding" in name:
                continue  # already initialized
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Token indices of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Create padding mask
        mask = (x != 0)  # (batch, seq_len)

        # Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # CNN expects (batch, channels, seq_len)
        embedded_t = embedded.transpose(1, 2)

        # Multi-scale CNN → trim to input seq_len → concat
        seq_len = embedded_t.size(2)
        conv_outputs = []
        for conv in self.convs:
            c = conv(embedded_t)  # (batch, num_filters, seq_len±1)
            # Trim to original seq_len (even kernels may add +1)
            c = c[:, :, :seq_len]
            conv_outputs.append(c)

        # Concatenate along channel dim: (batch, cnn_out_dim, seq_len)
        cnn_out = torch.cat(conv_outputs, dim=1)

        # Back to (batch, seq_len, cnn_out_dim) for LSTM
        cnn_out = cnn_out.transpose(1, 2)

        # Bi-LSTM
        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, lstm_out_dim)

        # Attention
        context = self.attention(lstm_out, mask)  # (batch, lstm_out_dim)

        # Classification
        logits = self.classifier(context)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# ===========================================================================
# Factory
# ===========================================================================

def build_text_model(
    vocab_size: int = 30000,
    num_classes: int = 2,
) -> TextCNN_BiLSTM:
    """Factory function to create the text classification model."""
    model = TextCNN_BiLSTM(vocab_size=vocab_size, num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TextCNN_BiLSTM created: {total_params:,} params ({trainable:,} trainable)")
    return model


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build vocab from sample data
    sample_texts = [
        "This is a perfectly safe sentence about coding.",
        "Beautiful sunset over the mountains today.",
        "This contains inappropriate adult content.",
    ]
    vocab = Vocabulary(max_vocab_size=1000, min_freq=1)
    vocab.build(sample_texts)
    print(f"Vocabulary size: {len(vocab)}")

    # Build model
    model = build_text_model(vocab_size=len(vocab))

    # Forward pass with dummy input
    encoded = [vocab.encode(t, max_length=32) for t in sample_texts]
    x = torch.tensor(encoded, dtype=torch.long)
    logits = model(x)
    proba = model.predict_proba(x)
    print(f"Input shape:  {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Proba shape:  {proba.shape}")
    print(f"Probabilities:\n{proba}")
