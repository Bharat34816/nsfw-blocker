import torch
import torch.nn as nn
import h5py
import onnx
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.efficientnet_model import EfficientNetB0
from models.text_model import TextCNN_BiLSTM

def save_as_h5(state_dict, path):
    """Save PyTorch state dict to an HDF5 file (portable weights)."""
    with h5py.File(path, 'w') as f:
        for key, value in state_dict.items():
            f.create_dataset(key, data=value.cpu().numpy())
    print(f"✅ Saved H5 Weights to: {path}")

def export_all_formats():
    """Export custom models to .pth, .onnx, and .h5 formats."""
    export_dir = Path("models/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Image Model (EfficientNet-B0)
    print("\n--- Exporting Image Model ---")
    image_model = EfficientNetB0(num_classes=2)
    image_model.eval()
    
    # Native PTH
    torch.save(image_model.state_dict(), export_dir / "image_model.pth")
    print(f"✅ Saved Native PTH to: {export_dir / 'image_model.pth'}")
    
    # ONNX (The gold standard for portability)
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = export_dir / "image_model.onnx"
    torch.onnx.export(
        image_model, dummy_input, onnx_path,
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Saved ONNX Model to: {onnx_path}")
    
    # H5 (HDF5 Weights)
    save_as_h5(image_model.state_dict(), export_dir / "image_model.h5")
    
    # 2. Text Model (CNN-BiLSTM)
    print("\n--- Exporting Text Model ---")
    text_model = TextCNN_BiLSTM(vocab_size=30000)
    text_model.eval()
    
    # Native PTH
    torch.save(text_model.state_dict(), export_dir / "text_model.pth")
    print(f"✅ Saved Native PTH to: {export_dir / 'text_model.pth'}")
    
    # ONNX
    dummy_text = torch.randint(0, 30000, (1, 128))
    onnx_text_path = export_dir / "text_model.onnx"
    torch.onnx.export(
        text_model, dummy_text, onnx_text_path,
        export_params=True, opset_version=14, # higher opset for RNN/LSTM support
        do_constant_folding=True,
        input_names=['tokens'], output_names=['logits'],
        dynamic_axes={'tokens': {0: 'batch_size', 1: 'seq_len'}, 'logits': {0: 'batch_size'}}
    )
    print(f"✅ Saved ONNX Model to: {onnx_text_path}")
    
    # H5
    save_as_h5(text_model.state_dict(), export_dir / "text_model.h5")

    print(f"\n🚀 All models exported successfully to: {export_dir.absolute()}")
    print("Formats: .pth (PyTorch), .onnx (Portable), .h5 (Weights)")

if __name__ == "__main__":
    export_all_formats()
