####### UTILITIES

import os 
import numpy as np
import torch
import random
from sklearn.metrics import confusion_matrix, roc_auc_score

# competition metric
def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score

# image paths
def get_train_file_path(image_id, CFG):
    return CFG['data_path'] + 'train/{}/{}/{}/{}.npy'.format(image_id[0], image_id[1], image_id[2], image_id)
def get_test_file_path(image_id, CFG):
    return CFG['data_path'] + 'test/{}/{}/{}/{}.npy'.format(image_id[0], image_id[1], image_id[2], image_id)

# random sequences
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

# device-aware printing
def smart_print(expression, accelerator = None):
    if accelerator is None:
        print(expression)
    else:
        accelerator.print(expression)

# randomness
def seed_everything(seed, accelerator = None):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    smart_print('- setting random seed to {}...'.format(seed), accelerator)
    
# torch random fix
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
# simple ensembles
def compute_blend(df, preds, blend, CFG, weights = None):
    
    if weights is None:
        weights = np.ones(len(preds)) / len(preds)
        
    if blend == 'amean':
        out = np.sum(df[preds].values * weights, axis = 1)
    elif blend == 'median':
        out = df[preds].median(axis = 1)
    elif blend == 'gmean':
        out = np.prod(np.power(df[preds].values, weights), axis = 1)
    elif blend == 'pmean':
        out = np.sum(np.power(df[preds].values, CFG['power']) * weights, axis = 1) ** (1 / CFG['power'])
    elif blend == 'rmean':
        out = np.sum(df[preds].rank(pct = True).values * weights, axis = 1)
    return out