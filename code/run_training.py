from utilities import *
from model import get_model
from train_fold import train_fold
from data import get_data
from plot_results import plot_results

import gc
import neptune
from accelerate import Accelerator, DistributedType
import torch
import pandas as pd
import numpy as np



def run_training(CFG, df):
    
    '''
    Run cross-validation loop
    '''
    
    # placeholder
    oof_score = []

    # cross-validation
    for fold in range(CFG['num_folds']):
        
        
        # initialize accelerator
        accelerator = Accelerator(device_placement = True,
                                  fp16             = CFG['use_fp16'],
                                  split_batches    = False)
        if CFG['num_devices'] == 1 and CFG['device'] == 'GPU':
            accelerator.state.device = torch.device('cuda:{}'.format(CFG['device_index']))

        # feedback
        accelerator.print('-' * 55)
        accelerator.print('FOLD {:d}/{:d}'.format(fold + 1, CFG['num_folds']))    
        accelerator.print('-' * 55) 

        # get model
        model = get_model(CFG, pretrained = CFG['pretrained'])

        # get data
        df_trn, df_val = get_data(df, fold, CFG, accelerator)  

        # run single fold
        trn_losses, val_losses, val_scores = train_fold(fold        = fold, 
                                                        df_trn      = df_trn,
                                                        df_val      = df_val, 
                                                        CFG         = CFG, 
                                                        model       = model, 
                                                        accelerator = accelerator)
        oof_score.append(np.max(val_scores))
        
        # feedback
        accelerator.print('-' * 55)
        accelerator.print('Best: score = {:.4f} (epoch {})'.format(
            np.max(val_scores), np.argmax(val_scores) + 1))
        accelerator.print('-' * 55)

        # plot loss dynamics
        if accelerator.is_local_main_process:
            plot_results(trn_losses, val_losses, val_scores, fold, CFG)

        # send weights to neptunes
        if CFG['tracking'] and accelerator.is_local_main_process:
            neptune.send_artifact(CFG['out_path'] + 'weights_fold{}.pth'.format(fold))
            
        # clear memory
        del accelerator
        gc.collect()
        
            
    # feedback
    print('')
    print('-' * 55)
    print('Mean OOF score = {:.4f}'.format(np.mean(oof_score)))
    print('-' * 55)
    if CFG['tracking']:
        neptune.send_metric('oof_score', np.mean(oof_score))