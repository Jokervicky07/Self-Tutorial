import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean', num_classes: int = 1):
        """
        Initialize the FocalLoss class.

        Args:
            alpha (float or list): Weighting factor for the focal loss as [alpha, 1-alpha, 1-alpha, ...]. 
                                If a list, it should have the same length as the number of classes.
            gamma (float): Focusing parameter for the focal loss.
            reduction (str): Reduction method. Options are 'none', 'mean', or 'sum'.
        """
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes, "Alpha list must have the same length as the number of classes."
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha > 0 and alpha < 1, "Alpha must be a float between 0 and 1."
            self.alpha = torch.Tensor([alpha] + [1 - alpha] * (num_classes - 1))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for focal loss.

        Args:
            input (tensor): Input tensor, shape (N, C) where N is the batch size and C is the number of classes.
            target (tensor): Target tensor, shape (N) where N is the batch size.

        Returns:
            tensor: Focal loss.
        """

        """
        \[
        \mathrm{FL}(p_t) = -\,(1 - p_t)^{r}\,\log\bigl(p_t\bigr)
        \]
        """
        # Apply softmax to get probabilities
        probs = F.softmax(input, dim=1).gather(1, target.view(-1, 1)).squeeze(1)
        
        # Compute the focal loss
        log_probs = F.log_softmax(input, dim=1).gather(1, target.view(-1, 1)).squeeze(1)
        alpha_t  = self.alpha[target].to(input.device)

        focal_loss = -alpha_t * ((1 - probs) ** self.gamma) * log_probs

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss