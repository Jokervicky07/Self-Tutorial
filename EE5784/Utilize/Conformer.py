import torch.nn as nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_mult=4, dropout=0.1):
        super(FeedForward, self).__init__()

        self.ffn = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, ff_mult * d_model),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(ff_mult * d_model, d_model))
    def forward(self, x):
        return self.ffn(x)

# https://arxiv.org/pdf/2005.08100
class ConformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, expansion_factor=2, ff_mult=4, conv_kernel=15, dropout=0.1):
        super(ConformerLayer, self).__init__()

        self.ffn1 = FeedForward(d_model, ff_mult, dropout)
        
        self.ln_mha = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=False)

        padding = conv_kernel // 2
        # input: (length, batch size, d_model), but the input dimension of Conv1d is (batch size, d_model, length)
        # so we need to rearrange the input tensor
        inner_dim = d_model * expansion_factor
        self.conv = nn.Sequential(nn.LayerNorm(d_model),
                                  Rearrange('l b d -> b d l'),
                                  nn.Conv1d(in_channels=d_model,out_channels=2*inner_dim,kernel_size=1,bias=False),
                                  nn.GLU(dim=1),
                                  nn.Conv1d(in_channels=inner_dim,out_channels=inner_dim,kernel_size=conv_kernel,
                                            padding=padding,groups=inner_dim,bias=False),
                                  nn.BatchNorm1d(inner_dim),
                                  nn.SiLU(),
                                  nn.Conv1d(in_channels=inner_dim,out_channels=d_model,kernel_size=1,bias=False),
                                  Rearrange('b d l -> l b d'),
                                  nn.Dropout(dropout))
        
        self.ffn2 = FeedForward(d_model, ff_mult, dropout)
        
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x):
        # Feed_Forward module
        # (length, batch size, d_model) -> (length, batch size, d_model)
        x = x + 0.5 * self.ffn1(x)

        # Multi-head attention module
        # (length, batch size, d_model) -> (length, batch size, d_model)
        x = self.ln_mha(x)
        y, _ = self.mha(x,x,x)
        x = x + y

        # Convolution module
        # (length, batch size, d_model) -> (length, batch size, d_model)
        x = x + self.conv(x)

        # Feed-forward module
        # (length, batch size, d_model) -> (length, batch size, d_model)
        x = x + 0.5 * self.ffn2(x)
        
        # Layer normalization
        x = self.ln_final(x)

        return x
    

class Conformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layer, ff_mult=4, conv_kernel=31, dropout=0.1):
        super(Conformer, self).__init__()

        self.layers = nn.ModuleList([])
        for i in range(num_layer):
            self.layers.append(ConformerLayer(d_model, n_heads, ff_mult, conv_kernel, dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x