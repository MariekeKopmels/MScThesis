import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        # Flatten the inputs to 2D tensors (batch_size * height * width, channels)
        prediction = prediction.view(-1, prediction.size(1))
        target = target.view(-1, target.size(1))

        # Calculate intersection and union
        intersection = torch.sum(prediction * target, dim=0)
        union = torch.sum(prediction + target, dim=0) - intersection

        # Calculate IoU for each channel
        iou = (intersection + self.smooth) / (union + self.smooth)

        # Average IoU across all channels
        mean_iou = torch.mean(iou)

        # Return 1 - mean IoU as the loss (to be minimized)
        loss = 1 - mean_iou

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction, target):
        # Calculate binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')

        # Calculate the modulating factor (alpha * target + (1 - alpha) * (1 - target))^gamma
        modulating_factor = (self.alpha * target + (1 - self.alpha) * (1 - target)).pow(self.gamma)

        # Calculate the focal loss
        focal_loss = bce_loss * modulating_factor

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
""" The loss function used for the multi task model. Combines the loss from the violence and skin colour heads.
"""
class MultiTaskLoss(nn.Module):
    def __init__(self, violence_weight=1.0, skincolour_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.violence_weight = violence_weight
        self.skincolour_weight = skincolour_weight
        self.violence_loss_function = nn.BCEWithLogitsLoss()
        self.skincolour_loss_function = nn.CrossEntropyLoss()
        
    def forward(self, violence_outputs, skincolour_outputs, violence_targets, skincolour_targets):
        # Calculate the individual head's losses
        violence_loss = self.violence_loss_function(violence_outputs, violence_targets)
        skincolour_loss = self.skincolour_loss_function(skincolour_outputs, skincolour_targets)
        
        # Combine the losses, using their weight
        combined_loss = self.violence_weight*violence_loss + self.skincolour_weight*skincolour_loss
        
        return combined_loss