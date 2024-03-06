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
        
""" A custom weighted MSE loss, due to the imbalance in target classes. 
    The weights defined should be in a torch tensor in the format
    [weight_for_target=1, weight_for_target=2, weight_for_target=3, weight_for_target=4, weight_for_target=5]
"""
class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        mse_loss = nn.MSELoss(reduction='none')(outputs, targets)
        
        if self.weights is not None:
            weights = torch.tensor([self.weights[int(target)-1] for target in targets], dtype=torch.float32)
            mse_loss = mse_loss * weights

        return mse_loss.mean()