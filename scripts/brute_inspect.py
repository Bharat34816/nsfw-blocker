import os
import sys
import pickle
from pathlib import Path

# Disable heavy logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def brute_force_inspect():
    keras_path = Path("models/external/text_model.keras")
    tokenizer_path = Path("models/external/tokenizer.pickle")
    if not tokenizer_path.exists():
        tokenizer_path = Path("models/external/tokenizer.pkl")
        
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    try:
        model = tf.keras.models.load_model(keras_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        print(f"Load Error: {e}")
        return

    test_text = "fuck you bastard"
    print(f"Testing string: '{test_text}'")
    
    configs = [
        (50, 'pre'), (50, 'post'),
        (256, 'pre'), (256, 'post'),
        (100, 'pre'), (100, 'post')
    ]
    
    print(f"{'Maxlen':<7} | {'Padding':<8} | {'Output Vector'}")
    print("-" * 40)
    for maxlen, padding in configs:
        try:
            seq = tokenizer.texts_to_sequences([test_text.lower()])
            padded = pad_sequences(seq, maxlen=maxlen, padding=padding, truncating=padding)
            pred = model.predict(padded, verbose=0)
            print(f"{maxlen:<7} | {padding:<8} | {pred[0]}")
        except Exception as e:
            print(f"{maxlen:<7} | {padding:<8} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    brute_force_inspect()
