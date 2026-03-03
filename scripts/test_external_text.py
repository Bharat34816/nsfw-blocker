import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.predictor import NSFWPredictor

def verify_external_model():
    print("--- Verifying External Text Model Integration ---")
    
    # Initialize predictor
    # It should automatically detect the files in models/external/
    predictor = NSFWPredictor()
    
    if predictor._has_external_text:
        print("✅ SUCCESS: External text model files detected.")
    else:
        print("❌ FAILURE: External text model files NOT detected.")
        return

    # Test prediction
    test_texts = [
        "This is a safe sentence.",
        "fuck you",
        "bastard"
    ]
    
    print("\nRunning test predictions...")
    for text in test_texts:
        result = predictor.predict_text(text)
        print(f"\nText: '{text}'")
        print(f"Result: {result.prediction}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Model Used: {result.details.get('model')}")
        print(f"Model Score: {result.details.get('model_score')}")
        print(f"Keyword Score: {result.details.get('keyword_score')}")

if __name__ == "__main__":
    predictor = NSFWPredictor()
    text = "bastard"
    result = predictor.predict_text(text)
    print(f"\n--- Focused Test ---")
    print(f"Text: '{text}'")
    print(f"Result: {result.prediction}")
    print(f"Model: {result.details.get('model')}")
    print(f"Model Score: {result.details.get('model_score')}")
    print(f"Final Score: {result.nsfw_score:.2f}")
