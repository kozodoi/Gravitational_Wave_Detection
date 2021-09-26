from utilities import *
from model import get_model
from data import get_data, get_loaders
from augmentations import get_augs
from test_epoch import test_epoch

import gc
import neptune
from accelerate import Accelerator, DistributedType
import pandas as pd
import numpy as np



def run_inference(df, df_test, CFG):
    
    '''
    Run inference loop
    '''
        
    # placeholders
    oof = None
    sub = None
    
    # inference
    for fold in range(CFG['num_folds']):
        
        # initialize accelerator
        accelerator = Accelerator(device_placement = True,
                                  fp16             = CFG['use_fp16'],
                                  split_batches    = False)
        if CFG['device'] == 'GPU':
            accelerator.state.device = torch.device('cuda:{}'.format(CFG['device_index']))

        # feedback
        accelerator.print('-' * 55)
        accelerator.print('FOLD {:d}/{:d}'.format(fold + 1, CFG['num_folds']))    
        accelerator.print('-' * 55)   
        
        # get data
        df_trn, df_val = get_data(df, fold, CFG, accelerator, silent = True)  

        # get test loader
        _, val_loader  = get_loaders(df_trn, df_val,  CFG, accelerator, labeled = False, silent = True) 
        _, test_loader = get_loaders(df_trn, df_test, CFG, accelerator, labeled = False, silent = True) 
        
        # prepare model
        model = get_model(CFG, pretrained = CFG['out_path'] + 'weights_fold{}.pth'.format(int(fold)))
        
        # handle device placement
        model, val_loader, test_loader = accelerator.prepare(model, val_loader, test_loader)
        
        # inference for validation data
        if CFG['predict_oof']:
                        
            # produce OOF preds
            val_preds = test_epoch(loader      = val_loader, 
                                   model       = model,
                                   CFG         = CFG,
                                   accelerator = accelerator,
                                   num_tta     = CFG['num_tta'])
            
            # store OOF preds
            val_preds_df = pd.DataFrame(val_preds, columns = ['pred'])
            val_preds_df = pd.concat([df_val, val_preds_df], axis = 1)
            oof          = pd.concat([oof,    val_preds_df], axis = 0).reset_index(drop = True)
                    
        # inference for test data
        if CFG['predict_test']:
            
            # produce test preds
            test_preds = test_epoch(loader      = test_loader, 
                                    model       = model,
                                    CFG         = CFG,
                                    accelerator = accelerator,
                                    num_tta     = CFG['num_tta'])
        
            # store test preds
            test_preds_df = pd.DataFrame(test_preds, columns = ['pred_fold{}'.format(int(fold))])
            sub           = pd.concat([sub, test_preds_df], axis = 1)
            
        # clear memory
        del model, val_loader, test_loader
        del accelerator
        gc.collect()
        
    # export OOF preds
    if CFG['predict_oof']:
        oof.to_csv(CFG['out_path'] + 'oof.csv', index = False)
        if CFG['tracking']:
            neptune.send_artifact(CFG['out_path'] + 'oof.csv')
            
    # export test preds
    if CFG['predict_test']:
        sub = pd.concat([df_test['id'], sub], axis = 1)
        sub.to_csv(CFG['out_path'] + 'submission.csv', index = False)
        if CFG['tracking']:
            neptune.send_artifact(CFG['out_path'] + 'submission.csv')