import pickle
from pathlib import Path

def check_vocab():
    tokenizer_path = Path("models/external/tokenizer.pickle")
    if not tokenizer_path.exists():
        tokenizer_path = Path("models/external/tokenizer.pkl")
        
    if not tokenizer_path.exists():
        print("Tokenizer not found")
        return

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    words_to_check = ["fuck", "bastard", "porn", "sex", "safe", "hello"]
    print(f"{'Word':<10} | {'Index'}")
    print("-" * 20)
    for word in words_to_check:
        idx = tokenizer.word_index.get(word, "NOT FOUND")
        print(f"{word:<10} | {idx}")

if __name__ == "__main__":
    check_vocab()
