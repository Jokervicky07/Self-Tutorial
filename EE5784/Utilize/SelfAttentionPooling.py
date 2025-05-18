import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionPooling(nn.Module):
    def __init__(self, d_model, attn_dim=None):
        super(SelfAttentionPooling, self).__init__()

        # if attn_dim is None, use d_model
        self.attn_dim = attn_dim or d_model
        self.linear = nn.Linear(d_model, self.attn_dim, bias=True)
        self.v = nn.Linear(self.attn_dim, 1, bias=False)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # u = tanh(x @ W + b)  -> (batch, seq_len, attn_dim)
        # attn_score = u @ v -> (batch, seq_len, 1)

        # activate: attn_score = tanh(x * W + b)v
        # attn-score: (batch_size, seq_len, 1)
        u = torch.tanh(self.linear(x))
        attn_score = self.v(u)
        # attn_score: (batch_size, seq_len)
        attn_score = attn_score.squeeze(-1)

        # alpha = exp(attn_score) / sum(exp(attn_score))
        # alpha: (batch_size, seq_len, 1)
        alpha = F.softmax(attn_score, dim=1).unsqueeze(-1)

        return torch.sum(x * alpha, dim=1), alpha