import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, num_features, vocab_size, num_heads=16, p=0.1):
        super().__init__()
        self.multihead_attention    = nn.MultiheadAttention(num_features, num_heads, p, batch_first=True)
        self.linear                 = nn.Linear(num_features * 2, vocab_size + 1)

    def forward(self, hq, hk, hv):
        x, _    = self.multihead_attention(hq, hk, hv)
        x       = torch.concat([x, hq], dim=2)
        x       = self.linear(x)
        return x