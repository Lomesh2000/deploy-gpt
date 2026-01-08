import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerBlock

class GPT(nn.Module):

    def __init__(self, n_layers, vocab_size, embed_size, block_size, num_head, device):
        super().__init__()
        self.device = device
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding_table = nn.Embedding(block_size, embed_size)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_size, num_head) for _ in range(n_layers)])

        self.layer_norm_final = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None, past_key_values=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        positional_embeddings = self.positional_embedding_table(torch.arange(T, device = self.device))       
        x = token_embeddings + positional_embeddings

        new_pkvs = []

        for i , block in enumerate(self.transformer_blocks):
            pkv = past_key_values[i]
            x, next_kv = block(x, pkv)
            new_pkvs.append(next_kv)

        x = self.layer_norm_final(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = target.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss, new_pkvs    

    def generate(self, idx, max_new_tokens):
        pkvs = None

        for _ in range(max_new_tokens):

            if pkvs is None:
                idx_cond = idx[:, -self.block_size:]

            else:
                idx_cond = idx[:, -1:]

        logits, _, pkvs = self(idx_cond, past_key_values=pkvs)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=-1)

        if idx.shape[1] > self.block_size:
            idx = idx[:, -self.block_size:]

        return idx            