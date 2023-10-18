import random
import yaml
from pathlib import Path

import numpy as np
import torch

# Load yaml config file
def load_config(path: Path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
