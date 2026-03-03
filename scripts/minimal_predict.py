import os
import sys
import pickle
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def minimal_predict():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    keras_path = Path("models/external/text_model.keras")
    tokenizer_path = Path("models/external/tokenizer.pickle")
    if not tokenizer_path.exists():
        tokenizer_path = Path("models/external/tokenizer.pkl")

    model = tf.keras.models.load_model(keras_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    test_text = "fuck you bastard porn"
    seq = tokenizer.texts_to_sequences([test_text.lower()])
    
    # Try maxlen 50, padding post
    padded = pad_sequences(seq, maxlen=50, padding='post')
    pred = model.predict(padded, verbose=0)
    
    print(f"Text: '{test_text}'")
    print(f"Output Vector: {pred[0]}")

if __name__ == "__main__":
    minimal_predict()
