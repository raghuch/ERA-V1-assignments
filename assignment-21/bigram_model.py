import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import *

from attention import SelfAttentionHead, MultiHeadAttention, FeedForwardNet, DecoderBlock


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, num_heads, n_layers) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(n_embed, num_heads, block_size=block_size) for _ in range(n_layers)] )
        self.ln_final = nn.LayerNorm(n_embed)

        ## self.sa_head = SelfAttentionHead(vocab_size, n_embed, block_size)
        # self.sa_heads = MultiHeadAttention(num_heads=4, head_size=n_embed//4, n_embed=n_embed, block_size=block_size)
        # self.ffn = FeedForwardNet(n_embed, dropout=0.2)

        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):

        # idx and targets both are tensors of shape (B, T) -> B = batch_sz, T = seq_len ("time steps", here 8)
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx) # (B, T, C)  C = "channels", here vocab_size or embedding dim for each token
        pos_embed = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)  C = "channels", here vocab_size or embedding dim for each token
        x_in = tok_embed + pos_embed
        # x_in = self.sa_heads(x_in)
        # x_in = self.ffn(x_in)
        x_in = self.ln_final(self.decoder_blocks(x_in))
        logits = self.lm_head(x_in) # (B, T, C)  C = "channels", here vocab_size or embedding dim for each token

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Cross entropy requires the 2nd param to be C "channels"
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), ignore_index=0)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) shaped array of indices in current context
        for _ in range(max_new_tokens):
            #limit input idx to last "block size" tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax for probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            #sample from distribudion
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #append sampled index to running sequence idx
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)

        return idx
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



if __name__ == "__main__":
    from data_utils import *
    xb, yb = get_random_batch('train')
    xb = xb.to(device)
    yb = yb.to(device)

    m = BigramLanguageModel(vocab_size=65, n_embed=n_embed, block_size=BLOCK_SIZE, num_heads=n_head, n_layers=n_layer).to(device)
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)
