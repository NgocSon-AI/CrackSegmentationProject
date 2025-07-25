import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice_Coeff(nn.Module):
    def __init__(self, smooth=1.0, threshold=0.5):
        super(Dice_Coeff, self).__init__()
        self.smooth = smooth
        self.threshold = threshold

    def forward(self, inputs, targets):
        # Apply threshold to convert probabilities to binary predictions
        inputs = (inputs > self.threshold).float()
        targets = targets.float()

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return dice


class IoU(nn.Module):
    def __init__(self, smooth=1.0, threshold=0.5):
        super(IoU, self).__init__()
        self.smooth = smooth
        self.threshold = threshold

    def forward(self, inputs, targets):
        inputs = (inputs > self.threshold).float()
        targets = targets.float()

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou
