import os
import sys
import pickle
from pathlib import Path

# Disable heavy logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def check_raw_output():
    keras_path = Path("models/external/text_model.keras")
    tokenizer_path = Path("models/external/tokenizer.pickle")
    if not tokenizer_path.exists():
        tokenizer_path = Path("models/external/tokenizer.pkl")
        
    if not keras_path.exists() or not tokenizer_path.exists():
        print("Model or Tokenizer not found")
        return

    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    model = tf.keras.models.load_model(keras_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    test_texts = [
        "This is a safe sentence",
        "fuck you",
        "bastard",
        "pornographic content",
        "hello world"
    ]
    
    print(f"{'Text':<25} | {'Raw Output Vector'}")
    print("-" * 50)
    for text in test_cases:
        seq = tokenizer.texts_to_sequences([text.lower()])
        padded = pad_sequences(seq, maxlen=256, padding='post')
        pred = model.predict(padded, verbose=0)
        print(f"{text:<25} | {pred[0]}")

if __name__ == "__main__":
    test_cases = [
        "This is a safe sentence",
        "fuck you",
        "bastard",
        "pornographic content",
        "hello world"
    ]
    check_raw_output()
