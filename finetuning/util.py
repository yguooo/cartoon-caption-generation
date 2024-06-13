import torch 
import random 
import numpy as np

def seed_everything(seed=2024):
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # If using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
        # The following two lines ensure deterministic behavior but may impact performance:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False