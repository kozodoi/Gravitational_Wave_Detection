from utilities import *
from data import get_loaders
from optimizers import get_optimizer
from schedulers import get_scheduler
from losses import get_losses
from train_epoch import train_epoch
from valid_epoch import valid_epoch

import time
import gc
import neptune
from accelerate import Accelerator, DistributedType



def train_fold(fold, df_trn, df_val, CFG, model, accelerator):
    
    '''
    Run training and validation on a single fold
    '''

    ##### PREPARATIONS
    
    # reset seed
    seed_everything(CFG['seed'] + fold, accelerator)
    
    # get data loaders
    trn_loader, val_loader = get_loaders(df_trn, df_val, CFG, accelerator)
        
    # get optimizer 
    optimizer = get_optimizer(CFG, model)
    
    # handle device placement
    model, optimizer, trn_loader, val_loader = accelerator.prepare(model, optimizer, trn_loader, val_loader)
    
    # get scheduler
    scheduler = get_scheduler(CFG, optimizer)
    
    # get losses
    trn_criterion, val_criterion = get_losses(CFG, accelerator)
        
    # placeholders
    trn_losses = []
    val_losses = []
    val_scores = []
    lrs        = []

    
    ##### TRAINING AND INFERENCE

    for epoch in range(CFG['num_epochs']):
                        
        # timer
        epoch_start = time.time()
        
        # training
        accelerator.wait_for_everyone()
        trn_loss = train_epoch(loader      = trn_loader, 
                               model       = model, 
                               optimizer   = optimizer, 
                               scheduler   = scheduler,
                               criterion   = trn_criterion, 
                               accelerator = accelerator,
                               epoch       = epoch,
                               CFG         = CFG)

        # inference with no TTA
        accelerator.wait_for_everyone()
        val_loss, val_preds, val_labels = valid_epoch(loader      = val_loader,
                                                      model       = model, 
                                                      criterion   = val_criterion, 
                                                      accelerator = accelerator,
                                                      CFG         = CFG,
                                                      num_tta     = 1)

        # save LR and losses
        accelerator.wait_for_everyone()
        lrs.append(scheduler.state_dict()['_last_lr'][0])
        trn_losses.append(trn_loss / len(df_trn) * CFG['num_devices'])
        val_losses.append(val_loss / len(df_val) * CFG['num_devices'])        
        val_scores.append(get_score(val_labels.astype('int'), val_preds))
        
        # feedback
        accelerator.wait_for_everyone()
        accelerator.print('-- epoch {}/{} | lr = {:.6f} | trn_loss = {:.4f} | val_loss = {:.4f} | val_auc = {:.4f} | {:.2f} min'.format(
            epoch + 1, CFG['num_epochs'], lrs[epoch],
            trn_losses[epoch], val_losses[epoch], val_scores[epoch],
            (time.time() - epoch_start) / 60))
        
        # send performance to Neptune
        if CFG['tracking'] and accelerator.is_local_main_process:
            neptune.send_metric('trn_lr_{}'.format(int(fold)),    lrs[epoch])
            neptune.send_metric('trn_loss_{}'.format(int(fold)),  trn_losses[epoch])
            neptune.send_metric('val_loss{}'.format(int(fold)),   val_losses[epoch])
            neptune.send_metric('val_score_{}'.format(int(fold)), val_scores[epoch])
            
        # export weights and save preds
        if val_scores[epoch] >= max(val_scores):
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), CFG['out_path'] + 'weights_fold{}.pth'.format(fold))
            
    # clear memory
    del model, optimizer, scheduler, trn_loader, val_loader, trn_criterion, val_criterion
            
    
    return trn_losses, val_losses, val_scores