import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class HeadAttention(nn.Module):

    def __init__(self, embed_size, head_size, block_size, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.head_size = head_size
        self.block_size = block_size
        self.dropout = dropout

        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None):

        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        if past_kv is not None:
            prev_k, prev_v = past_kv
            k = torch.cat([prev_k, k], dim=-1)
            v = torch.cat([prev_v, v], dim=-1)    

        if k.shape[1] > self.block_size:
            k = k[:, -self.block_size:]
            v = v[:, -self.block_size:]

        new_kv = (k, v)    

        att = Q @ K.transpose(-2, -1) * K.shape[-1]**(0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ V
        return out, new_kv


class MultiHeadAttention(nn.module):

    def __init__(self, num_head, head_size, embed_size, dropout):
        super().__init__()
        self.num_head = num_head
        self.head_size = head_size
        self.embed_size = embed_size

        self.heads = nn.ModuleList([HeadAttention(self.head_size) for _ in range(self.num_head)])
        self.proj = nn.Linear(head_size * num_head, self.embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_key_values=None):
        out_list = []
        new_kvs  = [] 

        for i, head in self.heads:
            past_kv = past_key_values[i]
            out, new_kv = head(x, past_kv)
            out_list.append(out)
            new_kvs.append(new_kv)

        out = torch.cat(out_list, dim=-1)
        out = self.dropout(self.proj(out))
        return out, new_kvs
        


