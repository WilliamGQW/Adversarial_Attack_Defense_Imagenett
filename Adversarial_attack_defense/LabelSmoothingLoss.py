import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

# Usually we predict 1 for the true class and 0 for the rest of them
# however, for label smoothing we predict (1-smoothing_value) for 
# the true class and smoothing_value for the rest of them.

# Thus the new CrossEntropy loss function with label smoothing is:
# loss = (1-smoothing_value)*CrossEntropyLoss(i) +  smoothing_value* mean(cross-entropy-loss(j))
# formula refered from this post:
# https://medium.com/towards-artificial-intelligence/how-to-use-label-smoothing-for-regularization-aa349f7f1dbb

class LabelSmoothingLoss(nn.Module):
    def __init__(self):
        super(LabelSmoothingLoss, self).__init__()
        
    def forward(self, image, target, smoothing_value=0.1):
        # compute standard cross-entropy-loss of j
        log_prediction = F.log_softmax(image, dim=-1) 
        smooth_loss = -log_prediction.mean(dim=-1)# compute mean
        # calculate the cross-entropy-loss of true class
        target = target.unsqueeze(1)
        nll = -log_prediction.gather(dim=-1, index=target) 
         # compute final cross-entropy loss with label smoothing
        loss = smoothing_value * smooth_loss + (1.0 - smoothing_value) * nll.squeeze(1)
        label_smoothing_loss = loss.mean()
        return label_smoothing_loss