import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, s=15.0):
        super(CosineLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.s = s

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # x_norm: (batch, in_features)
        # weight_norm.t(): (in_features, out_features)
        cos_theta = x_norm @ weight_norm.t()

        return self.s * cos_theta

