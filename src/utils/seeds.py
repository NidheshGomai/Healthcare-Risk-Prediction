"""
Global random seed management for reproducibility.
"""
import os
import random
import numpy as np
import torch

GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_generator(seed: int = GLOBAL_SEED) -> torch.Generator:
    """Get a PyTorch generator with the global seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


set_global_seed()