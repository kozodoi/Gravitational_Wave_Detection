import torch.optim as optim
from adamp import AdamP
from madgrad import MADGRAD



def get_optimizer(CFG, model):
    
    '''
    Get optimizer
    '''
                   
    # optimizer
    if CFG['optim'] == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr           = CFG['lr'], 
                               weight_decay = CFG['decay'])
    elif CFG['optim'] == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr           = CFG['lr'], 
                                weight_decay = CFG['decay'])
    elif CFG['optim'] == 'AdamP':
        optimizer = AdamP(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr           = CFG['lr'], 
                          weight_decay = CFG['decay'])
    elif CFG['optim'] == 'madgrad':
        optimizer = MADGRAD(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr           = CFG['lr'], 
                            weight_decay = CFG['decay']) 


    return optimizer