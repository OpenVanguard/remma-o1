import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embed size must be divisible by num_heads"

        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        N, seq_len, _ = x.shape
        
        # Split into multiple heads
        keys = self.keys(x).view(N, seq_len, self.num_heads, self.head_dim)
        queries = self.queries(x).view(N, seq_len, self.num_heads, self.head_dim)
        values = self.values(x).view(N, seq_len, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / math.sqrt(self.head_dim)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_len, self.embed_size)
        
        return self.fc_out(out)
    
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_size)
        )
        
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout):
        super().__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, ff_dim)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn = self.attention(x, mask)
        x = self.norm1(attn + x)
        x = self.dropout(x)
        
        ff = self.ff(x)
        x = self.norm2(ff + x)
        x = self.dropout(x)
        
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ff_dim, block_size, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(block_size, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)
        self.block_size = block_size
        
    def forward(self, x, targets=None):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        
        tok_emb = self.embed(x)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + pos_emb
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        logits = self.head(x)
        
        if targets is None:
            loss = None
        else:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
        return logits, loss

    # Key Differences from GPT-2
    # Positional Encoding: Using learned embeddings instead of sinusoidal
    # Normalization: LayerNorm placement differs from standard GPT
    # Initialization: No pretrained weights - you control all init
    # Architecture: Freedom to modify attention/FFN layers

