from utilities import *
from data import get_data, get_loaders
from augmentations import *

import neptune
from accelerate import Accelerator, DistributedType
import torch
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns



def sample_batch(CFG, df, sample_size = 5, batch_idx = 0):
    
    '''
    Display sample training and validation batch
    '''

    ##### PREPARATIONS

    # initialize accelerator
    accelerator = Accelerator(device_placement = True,
                              fp16             = CFG['use_fp16'],
                              split_batches    = False)
    accelerator.state.device = torch.device('cpu')

    # sample indices
    idx_start = batch_idx * sample_size
    idx_end   = (batch_idx + 1) * sample_size
    
    # data sample
    df_sample = pd.concat((df.iloc[idx_start:idx_end],
                           df.iloc[idx_start:idx_end]), axis = 0)
    df_sample['fold'] = np.concatenate((np.zeros(sample_size), np.ones(sample_size)))

    # get data
    df_trn, df_val = get_data(df          = df_sample, 
                              fold        = 0, 
                              CFG         = CFG, 
                              accelerator = accelerator, 
                              silent      = True, 
                              debug       = False)  

    # get data loaders
    trn_loader, val_loader = get_loaders(df_trn, df_val, CFG, accelerator, silent = True)

    # set seed
    seed_everything(CFG['seed'])


    ##### TRAIN IMAGES

    # display train images
    batch_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(trn_loader):
        
        # apply cutmix augmentation
        if CFG['cutmix'][0] > 0:
            mix_decision = 0 #np.random.rand(1)
            if mix_decision < CFG['cutmix'][0]:
                print('- applying cutmix...')
                inputs, _ = cutmix_fn(data   = inputs, 
                                      target = labels, 
                                      alpha  = CFG['cutmix'][1])

        # feedback
        inputs_shape = inputs.shape
        load_time    = time.time() - batch_time
        pixel_values = [torch.min(inputs).item(), torch.mean(inputs).item(), torch.max(inputs).item()]

        # examples
        fig = plt.figure(figsize = (20, 5))
        for i in range(sample_size):
            ax = fig.add_subplot(2, sample_size, i + 1, xticks = [], yticks = [])     
            plt.imshow(inputs[i].cpu().numpy()[0, :, :], cmap = 'gray')
            ax.set_title('{} (train)'.format(labels[i].cpu().numpy()), color = 'red')
        break

        
    ##### VALID IMAGES

    # display valid images
    batch_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(val_loader):
        
        # feedback
        print('- loading time: {:.4f} vs {:.4f} seconds'.format(load_time, (time.time() - batch_time)))
        print('- inputs shape: {} vs {}'.format(inputs_shape, inputs.shape))
        print('- pixel values: {:.2f} | {:.2f} | {:.2f} vs {:.2f} | {:.2f} | {:.2f}'.format(
                pixel_values[0], pixel_values[1], pixel_values[2],
                torch.min(inputs).item(), torch.mean(inputs).item(), torch.max(inputs).item()))

        # examples
        for i in range(sample_size):
            ax = fig.add_subplot(2, sample_size, sample_size + i + 1, xticks = [], yticks = [])     
            plt.imshow(inputs[i].cpu().numpy()[0, :, :], cmap = 'gray')
            ax.set_title('{} (valid)'.format(labels[i].cpu().numpy()), color = 'green')
        plt.savefig(CFG['out_path'] + 'fig_sample.png')
        break
        
    return (inputs, labels)
