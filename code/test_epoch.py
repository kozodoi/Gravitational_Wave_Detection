from augmentations import get_tta_flips

import numpy as np
import timm
from timm.utils import *
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType



def test_epoch(loader, 
               model, 
               CFG,
               accelerator,
               num_tta = None):
    
    '''
    Test epoch
    '''
    
    ##### PREPARATIONS
    
    # TTA options
    if num_tta is None:
        num_tta = CFG['num_tta']

    # switch regime
    model.eval()

    # placeholders
    PROBS = []
    
    # progress bar
    pbar = tqdm(range(len(loader)), disable = not accelerator.is_main_process)
    
    
    ##### INFERENCE LOOP
       
    # loop through batches
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):

            # preds placeholders
            logits = torch.zeros((inputs.shape[0], CFG['num_classes'] - 1), device = accelerator.device)
            probs  = torch.zeros((inputs.shape[0], CFG['num_classes'] - 1), device = accelerator.device)

            # compute predictions
            for tta_idx in range(num_tta): 
                preds   = model(get_tta_flips(inputs, tta_idx))
                logits += preds / num_tta
                probs  += preds.sigmoid() / num_tta

            # store predictions
            PROBS.append(accelerator.gather(probs).detach().cpu())
            
            # feedback
            pbar.update()

    # transform predictions
    return np.concatenate(PROBS)