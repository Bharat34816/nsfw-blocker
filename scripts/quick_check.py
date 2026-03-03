import os
import sys
from pathlib import Path
import logging

# Disable heavy logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from inference.predictor import NSFWPredictor

def quick_check():
    p = NSFWPredictor()
    t = "bastard"
    r = p.predict_text(t)
    print(f"--- VERIFICATION ---")
    print(f"Text: {t}")
    print(f"Model Info: {r.details.get('model')}")
    print(f"Result: {r.prediction}")
    print(f"Score: {r.nsfw_score}")

if __name__ == "__main__":
    quick_check()
