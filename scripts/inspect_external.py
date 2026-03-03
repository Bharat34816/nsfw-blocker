import os
import sys
import pickle
from pathlib import Path

# Disable heavy logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def inspect_external():
    keras_path = Path("models/external/text_model.keras")
    tokenizer_path = Path("models/external/tokenizer.pickle")
    if not tokenizer_path.exists():
        tokenizer_path = Path("models/external/tokenizer.pkl")
        
    print(f"--- Model Inspection ---")
    if keras_path.exists():
        import tensorflow as tf
        try:
            model = tf.keras.models.load_model(keras_path)
            print(f"Model Input Shape: {model.input_shape}")
            print(f"Model Output Shape: {model.output_shape}")
            # Try to see more info
            try:
                # Check for maxlen if it's stored in metadata
                pass
            except:
                pass
        except Exception as e:
            print(f"Model load error: {e}")
            
    print(f"\n--- Tokenizer Inspection ---")
    if tokenizer_path.exists():
        try:
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Check if it has texts_to_sequences
            print(f"Type: {type(tokenizer)}")
            if hasattr(tokenizer, 'word_index'):
                print(f"Vocab size: {len(tokenizer.word_index)}")
                # Show some words
                words = list(tokenizer.word_index.items())[:10]
                print(f"Sample words: {words}")
            
            # Check for other common attributes
            attrs = ['num_words', 'oov_token', 'filters']
            for attr in attrs:
                if hasattr(tokenizer, attr):
                    print(f"{attr}: {getattr(tokenizer, attr)}")
                    
        except Exception as e:
            print(f"Tokenizer load error: {e}")

if __name__ == "__main__":
    inspect_external()
