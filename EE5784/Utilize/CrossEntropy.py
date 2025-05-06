import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossEntropy(nn.Module):
    """
    CrossEntropy class for calculating cross-entropy loss.
    """

    def __init__(self, weight: Optional[torch.Tensor]=None, reduction: str='mean'):
        """
        Initialize the CrossEntropy class.
        Args:
            weight (Optional[torch.Tensor]): Weight tensor for the loss.
            reduction (str): Reduction method. Options are 'none', 'mean', or 'sum'.
        """
        super(CrossEntropy, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cross-entropy loss.

        Args:
            input (tensor): Input tensor, shape (N, C) where N is the batch size and C is the number of classes.
            target (tensor): Target tensor, shape (N) where N is the batch size.

        Returns:
            tensor: Cross-entropy loss.
        """
        logsoftmax_input = F.log_softmax(input, dim=1)
        
        """
        \[
        \ell(x, y)
        = L = [\,l_1, \dots, l_N\,]^\top,
        \quad
        l_n = -\,w_{y_n}\,x_{n, y_n},
        \quad
        w_c = \mathrm{weight}[c]\;\mathbf{1}\{c \neq \text{ignore\_index}\}\,.
        \]

        \[
        \ell(x, y) =
        \begin{cases}
        \displaystyle
        \frac{1}{\sum_{n=1}^N w_{y_n}}\sum_{n=1}^N l_n,
        &\text{if reduction = `mean'},\\[1ex]
        \displaystyle
        \sum_{n=1}^N l_n,
        &\text{if reduction = `sum'}.
        \end{cases}
        \]
        \]

        """
        # nll_loss = -logsoftmax_input.gather(1, target.view(-1, 1)).squeeze(1)
        return F.nll_loss(logsoftmax_input, target, weight=self.weight, reduction=self.reduction)
