import torch
import torch.nn as nn
import torch.nn.functional as F

class AMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.3):
        super(AMSoftmax, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.s = s
        self.m = m

    def forward(self, x, label=None):
        x_norm = F.normalize(x, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # x_norm: (batch, in_features)
        # weight_norm.t(): (in_features, out_features)
        cos_theta = x_norm @ weight_norm.t()
        
        # l_j = s * (cos_theta_j - 1(y=j)*m)
        if label is not None:
            phi = cos_theta.clone()
            phi[torch.arange(x.size(0)), label] -= self.m
            out = self.s * phi
        else:
            out = self.s * cos_theta

        return out

