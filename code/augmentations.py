import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch



def get_augs(CFG, p_aug = None):
    
    '''
    Get train and test augmentations
    '''

    # update epoch-based parameters
    if p_aug is None:
        p_aug = CFG['p_aug']
        
    # blur
    if CFG['blur_limit'] == 0:
        p_blur = 0
    else:
        p_blur = p_aug
        
    # normalization
    if CFG['normalize']:
        p_norm = 1
    else:
        p_norm = 0
    CFG['pixel_mean'] = 0
    CFG['pixel_std']  = 1
    
    # train augmentations
    train_augs = A.Compose([A.Transpose(p      = CFG['p_transpose']),
                            A.HorizontalFlip(p = CFG['p_flip']),
                            A.ShiftScaleRotate(p            = p_aug,
                                               shift_limit  = CFG['ssr'][0],
                                               scale_limit  = CFG['ssr'][1],
                                               rotate_limit = CFG['ssr'][2]),
                            A.HueSaturationValue(p               = p_aug,
                                                 hue_shift_limit = CFG['huesat'][0],
                                                 sat_shift_limit = CFG['huesat'][1],
                                                 val_shift_limit = CFG['huesat'][2]),
                            A.RandomBrightnessContrast(p                = p_aug,
                                                       brightness_limit = CFG['bricon'][0],
                                                       contrast_limit   = CFG['bricon'][1]),
                            A.OneOf([A.MotionBlur(blur_limit   = CFG['blur_limit']),
                                     A.GaussianBlur(blur_limit = CFG['blur_limit'])], 
                                     p                         = p_blur),
                            A.Normalize(p    = p_norm,
                                        mean = CFG['pixel_mean'],
                                        std  = CFG['pixel_std']),
                            ToTensorV2()
                           ])

    # valid augmentations
    valid_augs = A.Compose([A.Normalize(p    = p_norm,
                                        mean = CFG['pixel_mean'],
                                        std  = CFG['pixel_std']),
                            ToTensorV2()
                           ])
    
    # output
    return train_augs, valid_augs



####### TTA FLIPS
 
def get_tta_flips(img, i):
    
    '''
    Get TTA flips
    Based on https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
    '''

    if i >= 4:
        img = img.transpose(2, 3)
    if i % 4 == 0:
        return img
    elif i % 4 == 1:
        return img.flip(3)
    elif i % 4 == 2:
        return img.flip(2)
    elif i % 4 == 3:
        return img.flip(3).flip(2)
    
    
    
####### CUTMIX

def rand_bbox(size, lam):
    
    '''
    Random image box for cutmix
    '''

    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w   = np.int(W * cut_rat)
    cut_h   = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_fn(data, target, alpha):
    
    '''
    Cutmix augmentation
    '''

    indices       = torch.randperm(data.size(0))
    shuffled_data   = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets