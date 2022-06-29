import torch 
import random 
import numpy as np

import json
from types import SimpleNamespace
import logging
import os
import time

def get_config(file):
    with open(file, 'r') as f:
        return json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))
    
def set_all_seeds(seed_value):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False