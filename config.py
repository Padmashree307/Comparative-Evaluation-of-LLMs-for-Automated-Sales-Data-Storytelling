"""
Configuration for reproducible results
"""

import os
import numpy as np
import random

# ==================== RANDOM SEED CONFIGURATION ====================
RANDOM_SEED = 42

def set_seeds():
    """Set all random seeds for reproducibility."""
    # Python random
    random.seed(RANDOM_SEED)
    
    # NumPy random
    np.random.seed(RANDOM_SEED)
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    
    print(f"✓ Random seeds set to: {RANDOM_SEED}")

# ==================== LLM CONFIGURATION ====================
LLM_CONFIG = {
    'temperature': 0,  # 0 = deterministic, no randomness
    'max_tokens': 2000,
    'top_p': 0.9,
}

# ==================== CLUSTERING CONFIGURATION ====================
KMEANS_CONFIG = {
    'random_state': RANDOM_SEED,
    'n_init': 10,
    'max_iter': 300,
}

# ==================== OTHER CONFIGURATION ====================
DATA_RANDOM_STATE = RANDOM_SEED
OUTPUT_DIR = "research_outputs"
