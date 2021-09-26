import torch
import torch.nn as nn
import torch.nn.functional as F


def get_losses(CFG, accelerator):
    
    '''
    Get loss function
    '''

    # define training loss
    if CFG['loss_fn'] == 'BCE':
        train_criterion = nn.BCEWithLogitsLoss()
    elif CFG['loss_fn'] == 'FC':
        train_criterion = FocalCosineLoss(device = accelerator.device)
    
    # define valid loss
    valid_criterion = nn.BCEWithLogitsLoss()

    return train_criterion, valid_criterion



class FocalCosineLoss(nn.Module):
    
    def __init__(self, device, alpha = 1, gamma = 2, xent = 0.1, reduction = 'mean'):
        super(FocalCosineLoss, self).__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.xent      = xent
        self.reduction = reduction
        self.device    = device
        self.y         = torch.Tensor([1])
        
    def forward(self, input, target):
        cent_loss   = nn.BCEWithLogitsLoss()(input, target).to(self.device)
        cosine_loss = F.cosine_embedding_loss(input.unsqueeze(0), 
                                              target.unsqueeze(0), 
                                              self.y.to(self.device), 
                                              reduction = self.reduction)
        pt          = torch.exp(-cent_loss)
        focal_loss  = self.alpha * (1-pt)**self.gamma * cent_loss

        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        
        return cosine_loss + self.xent * focal_loss


