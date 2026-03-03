import torch
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.efficientnet_model import EfficientNetB0
from models.text_model import TextCNN_BiLSTM

def save_initial_models():
    """Save an initial (randomly initialized) model for architecture verification."""
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Initializing Custom Models ---")
    
    # Image model
    image_model = EfficientNetB0(num_classes=2)
    image_path = checkpoint_dir / "image_model.pth"
    torch.save(image_model.state_dict(), image_path)
    print(f"✅ Saved initial Image Model to: {image_path}")
    
    # Text model
    text_model = TextCNN_BiLSTM(vocab_size=30000)
    text_path = checkpoint_dir / "text_model.pth"
    torch.save(text_model.state_dict(), text_path)
    print(f"✅ Saved initial Text Model to: {text_path}")
    
    print(f"\nNOTE: These models have random weights. They must be trained using ")
    print(f"training/train_image.py and training/train_text.py with real data.")

if __name__ == "__main__":
    save_initial_models()
