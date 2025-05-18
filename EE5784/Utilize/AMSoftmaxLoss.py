import torch
import torch.nn as nn
import torch.nn.functional as F

class AMSoftmaxLoss(nn.Module):
    def __init__(self, s=15.0, m=0.3):
        super(AMSoftmaxLoss, self).__init__()
        """
        margin=0.3, scale=15.0
        """
        self.s = s
        self.m = m

    def forward(self, logits, labels):
        """
        logits: (batch_size, num_classes) == s * cosine(theta), which is scaled
        labels: (batch_size,_)
        """

        phi = logits.clone()
        phi[torch.arange(logits.size(0)), labels] -= self.m * self.s

        # calculate the log sum exp
        # log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=1))
        # logits = logits - log_sum_exp
        lse = torch.logsumexp(logits, dim=1)
        logits = lse - phi[torch.arange(logits.size(0)), labels]

        return logits.mean()