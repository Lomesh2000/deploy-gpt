import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

class FeedForward(nn.Module):

    def __init__(self, embed_size, dropout):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(self, embed_size, num_head):
        super().__init__()
        head_size = embed_size // num_head

        self.selfAttention = MultiHeadAttention(embed_size, head_size)
        self.feedforward = FeedForward(embed_size)
        self.layerNorm1 = nn.LayerNorm(embed_size)
        self.layerNorm2 = nn.LayerNorm(embed_size)

    def forward(self, x, past_kv=None):
        attn_out, next_kv = self.selfAttention(self.layerNorm1(x), past_key_values=past_kv)
        x = x + attn_out
        x = x + self.feedforward(self.layerNorm2(x))
        return x, next_kv