from utilities import *
from data import get_data, get_loaders
from model import get_model
from augmentations import *

import neptune
from accelerate import Accelerator, DistributedType
import torch
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns



def plot_predictions(CFG, fold, sample_size = 5):
    
    '''
    Display right and wrong predictions
    '''

    ##### PREPARATIONS

    # initialize accelerator
    accelerator = Accelerator(device_placement = True,
                              fp16             = CFG['use_fp16'],
                              split_batches    = False)
    accelerator.state.device = torch.device('cpu')

    # data sample
    oof = pd.read_csv(CFG['out_path'] + 'oof.csv')
    oof['fold']      = 0
    oof['confidence'] = np.maximum(oof['pred'], 1 - oof['pred'])
    oof['pred_bin']  = np.round(oof['pred'])
    oof['correct']   = oof['pred_bin'] == oof['target']
    oof              = oof.sort_values(['target', 'confidence'], ascending = True)

    # split good and bad preds
    rights = oof.loc[oof['correct'] == True].groupby('target').tail(sample_size).reset_index(drop  = True)
    wrongs = oof.loc[oof['correct'] == False].groupby('target').tail(sample_size).reset_index(drop = True)

    # get data loaders
    _, right_loader = get_loaders(rights, rights, CFG, accelerator, silent = True)
    _, wrong_loader = get_loaders(wrongs, wrongs, CFG, accelerator, silent = True)
    
    # image grid
    fig = plt.figure(figsize = (20, 20))

    # right preds
    for batch_idx, (inputs, labels) in enumerate(right_loader):
        for i in range(inputs.shape[0]):
            ax = fig.add_subplot(2 * CFG['num_classes'], sample_size, i + 1, xticks = [], yticks = [])     
            plt.imshow(inputs[i].cpu().numpy()[0, :, :], cmap = 'gray')
            ax.set_title('{} [pred = {:.4f}]'.format(
                labels[i].numpy(), rights.iloc[i]['pred'], rights.iloc[i]['confidence']), color = 'green')

    # wrong preds
    for batch_idx, (inputs, labels) in enumerate(wrong_loader):
        for i in range(inputs.shape[0]):
            ax = fig.add_subplot(2 * CFG['num_classes'], sample_size, 2 * sample_size + i + 1, xticks = [], yticks = [])     
            plt.imshow(inputs[i].cpu().numpy()[0, :, :], cmap = 'gray')
            ax.set_title('{} [pred = {:.4f}]'.format(
                labels[i].numpy(), wrongs.iloc[i]['pred'], wrongs.iloc[i]['confidence']), color = 'red')

    # export
    plt.savefig(CFG['out_path'] + 'fig_errors.png')
    plt.show()