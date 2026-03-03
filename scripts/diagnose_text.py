import os
import sys
from pathlib import Path
import logging

# Disable heavy logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.predictor import NSFWPredictor

def diagnose():
    print("="*60)
    print("NSFW TEXT PREDICTION DIAGNOSTIC")
    print("="*60)
    
    p = NSFWPredictor()
    
    test_cases = [
        "This is a perfectly safe sentence for testing.",
        "I love coding in Python and building AI apps.",
        "fuck you",
        "you are a bastard",
        "p[o0]rn video",
        "sexy girl",
        "hot babe",
        "bastard",
        "Get out of here",
        "This is a pornographic website"
    ]
    
    print(f"{'Text Content':<35} | {'Source':<15} | {'Model':<6} | {'KW':<5} | {'Final':<5} | {'Result':<7}")
    print("-" * 95)
    
    for text in test_cases:
        result = p.predict_text(text)
        
        source = result.details.get('model', 'UNKNOWN')
        model_score = result.details.get('model_score')
        kw_score = result.details.get('keyword_score')
        
        # Format scores: handle None, round to 2 decimals
        m_str = f"{model_score:.2f}" if model_score is not None else "N/A"
        k_str = f"{kw_score:.2f}" if kw_score is not None else "N/A"
        f_str = f"{result.nsfw_score:.2f}"
        
        # Truncate text for display
        display_text = (text[:32] + '...') if len(text) > 32 else text
        
        print(f"{display_text:<35} | {source:<15} | {m_str:<6} | {k_str:<5} | {f_str:<5} | {result.prediction:<7}")

if __name__ == "__main__":
    diagnose()
