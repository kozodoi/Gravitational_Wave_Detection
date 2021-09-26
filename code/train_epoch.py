import timm
from timm.utils import *
from utilities import *
from augmentations import *

import torch
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType



def train_epoch(loader, 
                model, 
                optimizer, 
                scheduler, 
                criterion, 
                accelerator,
                epoch, 
                CFG):
        
    '''
    Training epoch
    '''
        
    ##### PREPARATIONS
       
    # switch regime
    model.train()

    # running loss
    trn_loss = AverageMeter()
    
    # loader length
    len_loader = CFG['max_batches'] if CFG['max_batches'] else len(loader) 

    # progress bar
    pbar = tqdm(range(len_loader), disable = not accelerator.is_main_process)

    
    ##### TRAINING LOOP
    
    # loop through batches
    for batch_idx, (inputs, labels) in enumerate(loader):
        
        # apply cutmix augmentation
        if CFG['cutmix'][0] > 0:
            mix_decision = np.random.rand(1)
            if mix_decision < CFG['cutmix'][0]:
                inputs, labels = cutmix_fn(data   = inputs, 
                                           target = labels, 
                                           alpha  = CFG['cutmix'][1])
        else:
            mix_decision = 0
        
        # label smoothing
        if CFG['smooth']:
            for image_idx, label in enumerate(labels):
                labels[image_idx] = torch.where(label == 1.0, 1.0 - CFG['smooth'], CFG['smooth'])

        # update scheduler on batch
        if CFG['upd_on_batch']:
            scheduler.step(epoch + batch_idx / len(loader))

        # passes and weight updates
        with torch.set_grad_enabled(True):
            
            # forward pass 
            logits = model(inputs)
            if (CFG['cutmix'][0] > 0) and (mix_decision < CFG['cutmix'][0]):
                loss = criterion(logits.view(-1), labels[0]) * labels[2] + criterion(logits.view(-1), labels[1]) * (1. - labels[2])
            else:
                loss = criterion(logits.view(-1), labels)
            loss = loss / CFG['accum_iter']
                
            # backward pass
            accelerator.backward(loss)
                
            # gradient clipping
            if CFG['grad_clip']:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), CFG['grad_clip'])

            # update weights
            if ((batch_idx + 1) % CFG['accum_iter'] == 0) or ((batch_idx + 1) == len_loader):
                optimizer.step()
                optimizer.zero_grad()

        # update loss
        trn_loss.update(loss.item() * CFG['accum_iter'], inputs.size(0))
        
        # feedback
        pbar.update()
        if CFG['batch_verbose']:
            if (batch_idx > 0) and (batch_idx % CFG['batch_verbose'] == 0):
                accelerator.print('-- batch {} | cur_loss = {:.6f}, avg_loss = {:.6f}'.format(
                    batch_idx, loss.item(), trn_loss.avg), CFG)
                
        # early stop
        if (batch_idx == len_loader):
            break
            
    # update scheduler on epoch
    if not CFG['upd_on_batch']:
        scheduler.step() 

    return trn_loss.sum