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
        return json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d)), file
    
def set_all_seeds(seed_value):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False

def prepare_output_directory(train_config, model_config):

    path = "tmp/" + model_config.model 
    if train_config.from_pretrained:
        path += "_pretrained"
    else:
        path += "_scratch"  

    if train_config.mask_inputs:
        path += "_masked"
    else:
        path += "_unmasked"
    
    if train_config.all_classes:
        path += "_all"
    else:
        path += "_binary"

    path += "_" + str(train_config.mask_mean)
    path += "_" + str(train_config.mask_std)

    try:
        path += "_" + str(model_config.block_size) 
        path += "_" + str(model_config.stride) 

        if model_config.full_mask_only:
            path += "_full"
        elif model_config.circular_mask:
            path += "_circular"
        else:
            path += "_balanced"
    except: pass

    path += "_" + time.strftime("%Y%m%d%H%M%S") 
    import os.path
    from os import path as p
    from random import randint

    while p.exists(path):
        path = path + "_" + str(randint(0, 10000))

    os.makedirs(path)

    logging.basicConfig(
        filename=path + '/output.log', 
        level=logging.INFO, format="[%(asctime)s][%(levelname)s]%(message)s", 
        datefmt='%Y-%m-%d %H:%M:%S'
        )

    logging.info(f"Train config: {train_config}")
    logging.info(f"Train config: {model_config}")

    os.makedirs(path + "/outputs")
        
    return path

def save_matrices(outputs, path): 

    names = ["base", "processed", "segmentation", "attribution", "labels", "pos_labels", "true_labels"]
    
    for name, output in zip(names, outputs):
        output = torch.cat(output, dim=0).cpu().numpy()

        logging.info(f"Saving {name} {output.shape}")

        with open(path + '/outputs/' + name + '.npy', 'wb') as f:
            np.save(f, output)



    
