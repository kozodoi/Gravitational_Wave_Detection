import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from nnAudio.Spectrogram import CQT1992v2
import librosa

import numpy as np
import pandas as pd

from scipy import signal, optimize
from timm.models.layers.conv2d_same import conv2d_same

from utilities import *
from augmentations import get_augs, CWT


class ImageData(Dataset):
    
    '''
    Image dataset class
    '''
    
    def __init__(self, 
                 df, 
                 transformation,
                 wave_transform_params,
                 channels  = 1,
                 labeled   = True,
                 transform = None):
        self.df             = df
        self.labeled        = labeled
        self.transformation = transformation
        self.transform      = transform
        self.channels       = channels
        self.wave_transform = CQT1992v2(**wave_transform_params)
        self.cwt            = CWT(fmin = 20, fmax = 500, hop_length = 8, dj = 0.125/8)
        self.bandpass       = signal.butter(4, [35, 500], btype = "bandpass", output = "sos", fs = 2048)
        
    def __len__(self):
        return len(self.df)
    
    def apply_wave_transform(self, waves, channels, transform):
        if channels == 1:
            waves = np.hstack(waves)
            waves = waves / np.max(waves)
            waves = torch.from_numpy(waves).float()
            image = transform(waves)
        elif channels == 3:
            w0 = waves[0, :] / np.max(waves[0, :])
            w1 = waves[1, :] / np.max(waves[1, :])
            w2 = waves[2, :] / np.max(waves[2, :])
            w0 = torch.from_numpy(w0).float()
            w1 = torch.from_numpy(w1).float()
            w2 = torch.from_numpy(w2).float()
            w0 = transform(w0)
            w1 = transform(w1)
            w2 = transform(w2)
            image = torch.cat((w0, w1, w2), dim = 0)
            image = torch.transpose(image, 0, 2)
        return image
    
    def apply_spec_transform(self, waves, channels):
        if channels == 1:
            waves = np.hstack(waves)
            waves = waves / np.max(waves)
            image = librosa.stft(waves)
            image = librosa.amplitude_to_db(abs(image))
            image = image / np.max(image)
            image = torch.from_numpy(image).float()
        elif channels == 3:
            w0 = waves[0, :] / np.max(waves[0, :])
            w1 = waves[1, :] / np.max(waves[1, :])
            w2 = waves[2, :] / np.max(waves[2, :])
            w0 = librosa.stft(w0)
            w1 = librosa.stft(w1)
            w2 = librosa.stft(w2)
            w0 = librosa.amplitude_to_db(abs(w0))
            w1 = librosa.amplitude_to_db(abs(w1))
            w2 = librosa.amplitude_to_db(abs(w2))
            w0 = w0 / np.max(w0)
            w1 = w1 / np.max(w1)
            w2 = w2 / np.max(w2)
            w0 = torch.from_numpy(w0).float()
            w1 = torch.from_numpy(w1).float()
            w2 = torch.from_numpy(w2).float()
            image = torch.stack((w0, w1, w2), dim = 0)
            image = torch.transpose(image, 0, 2)
        return image
    
    def apply_cwt_transform(self, waves, channels):
        
        if channels == 3:

            # tuckey
            waves = waves / np.max(waves)
            waves *= signal.tukey(4096, 0.2)

            # bandpass
            normalization = np.sqrt((500 - 35) / (2048 / 2))
            waves         = signal.sosfiltfilt(self.bandpass, waves) / normalization

            # CWT
            image = torch.tensor(waves, dtype = torch.float32).view(1, 3, 4096)
            image = self.cwt(image)
            image = torch.transpose(image[0], 0, 2)
            image = torch.transpose(image,    0, 1)
            image = image / torch.max(image)
            
        return image
        

    def __getitem__(self, idx):
        
        # import image
        file_path = self.df.loc[idx, 'file_path']
        image    = np.load(file_path)
        if image is None:
            raise FileNotFoundError(file_path)
                        
        # wave transformation
        if self.transformation == 'q':
            image = self.apply_wave_transform(image, self.channels, self.wave_transform)
            image = image.squeeze().numpy()      
        elif self.transformation == 's':
            image = self.apply_spec_transform(image, self.channels)
            image = image.squeeze().numpy()   
        elif self.transformation == 'cwt':
            image = self.apply_cwt_transform(image, self.channels)
            image = image.squeeze().numpy()   
                
        # augmentations
        if self.transform:
            image = self.transform(image = image)['image']

        # output
        if self.labeled:
            label = torch.tensor(self.df.loc[idx, 'target']).float()
            return image, label            
        return image            

    

def get_data(df, fold, CFG, accelerator, silent = False, debug = None):
    
    '''
    Get training and validation data
    '''

    # load splits
    df_train = df.loc[df.fold != fold].reset_index(drop = True)
    df_valid = df.loc[df.fold == fold].reset_index(drop = True)
    if not silent:
        accelerator.print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
        
    # subset for debug mode
    if debug is None:
        debig = CFG['debug']
    if debug:
        df_train = df_train.sample(CFG['batch_size'] * 10, random_state = CFG['seed']).reset_index(drop = True)
        df_valid = df_valid.sample(CFG['batch_size'] * 10, random_state = CFG['seed']).reset_index(drop = True)
        accelerator.print('- subsetting data for debug mode...')
        accelerator.print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
    
    return df_train, df_valid



def get_loaders(df_train, df_valid, CFG, accelerator, labeled = True, silent = False):
    
    '''
    Get training and validation dataloaders
    '''

    ##### DATASETS
    
    # wave transformation
    wave_transform_params = {"sr":              CFG['q_sr'], 
                             "fmin":            CFG['q_fmin'], 
                             "fmax":            CFG['q_fmax'], 
                             "hop_length":      CFG['q_hop'], 
                             "bins_per_octave": CFG['q_bins'],
                             'verbose':         False}
        
    # augmentations
    train_augs, valid_augs = get_augs(CFG, CFG['p_aug'])
    
    # datasets
    train_dataset = ImageData(df                    = df_train, 
                              transformation        = CFG['transform'],
                              wave_transform_params = wave_transform_params,
                              channels              = CFG['channels'],
                              transform             = train_augs,
                              labeled               = labeled)
    valid_dataset = ImageData(df                    = df_valid, 
                              transformation        = CFG['transform'],
                              wave_transform_params = wave_transform_params,
                              channels              = CFG['channels'],
                              transform             = valid_augs,
                              labeled               = labeled)

        
    ##### DATA LOADERS
    
    # data loaders
    train_loader = DataLoader(dataset        = train_dataset, 
                              batch_size     = CFG['batch_size'], 
                              shuffle          = True,
                              num_workers    = CFG['cpu_workers'],
                              drop_last      = False, 
                              worker_init_fn = worker_init_fn,
                              pin_memory     = False)
    valid_loader = DataLoader(dataset     = valid_dataset, 
                              batch_size  = CFG['valid_batch_size'], 
                              shuffle       = False,
                              num_workers = CFG['cpu_workers'],
                              drop_last   = False,
                              pin_memory  = False)
    
    # feedback
    if not silent:
        accelerator.print('-  p(augment): {}'.format(CFG['p_aug']))
        accelerator.print('-' * 55)
    
    return train_loader, valid_loader



