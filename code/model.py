####### MODEL PREP

from utilities import *
import timm
import torch
import torch.nn as nn
import gc

def get_model(CFG, pretrained = None):
    
    # pretrained weights
    if pretrained is None:
        pretrained = CFG['pretrained']
        
    # input channels
    num_channels = CFG['channels']

    # CNN part
    model = timm.create_model(model_name = CFG['backbone'], 
                              pretrained = False if not pretrained else True,
                              in_chans   = num_channels)

    # classifier part                            
    if 'efficient' in CFG['backbone']:
        model.classifier = nn.Linear(model.classifier.in_features, CFG['num_classes'] - 1)
    elif 'vit' in CFG['backbone'] or 'swin' in CFG['backbone']:
        model.head = nn.Linear(model.head.in_features, CFG['num_classes'] - 1)
    elif 'nfnet' in CFG['backbone']:
        model.head.fc = nn.Linear(model.head.fc.in_features, CFG['num_classes'] - 1)
    else:
        model.fc = nn.Linear(model.fc.in_features, CFG['num_classes'] - 1)
        
    # load pre-trained weights
    if pretrained:
        if pretrained != 'imagenet':
            model.load_state_dict(torch.load(pretrained, map_location = torch.device('cpu')))
            print('-- loaded custom weights')
            
    return model