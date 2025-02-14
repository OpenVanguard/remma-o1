import torch
import torch.nn as nn
import math
# Self-Attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size  # Size of the embedding vector
        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = embed_size // num_heads  # Dimension of each attention head

        # Ensure embed_size is divisible by num_heads
        assert self.head_dim * num_heads == embed_size, "Embed size must be divisible by num_heads"

        # Linear layers to project input to queries, keys, and values
        self.keys = nn.Linear(embed_size, embed_size)  # Linear layer for keys
        self.queries = nn.Linear(embed_size, embed_size)  # Linear layer for queries
        self.values = nn.Linear(embed_size, embed_size)  # Linear layer for values
        self.fc_out = nn.Linear(embed_size, embed_size)  # Linear layer for output

    def forward(self, x, mask=None):
        N, seq_len, _ = x.shape  # Batch size, sequence length, and embedding size
        
        # Split the embedding into multiple heads for multi-head attention
        keys = self.keys(x).view(N, seq_len, self.num_heads, self.head_dim)  # Project and reshape keys
        queries = self.queries(x).view(N, seq_len, self.num_heads, self.head_dim)  # Project and reshape queries
        values = self.values(x).view(N, seq_len, self.num_heads, self.head_dim)  # Project and reshape values
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / math.sqrt(self.head_dim)  # Compute attention scores
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))  # Apply mask to attention scores
        
        attention = torch.softmax(energy, dim=-1)  # Normalize attention scores
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_len, self.embed_size)  # Compute attention output and reshape
        
        return self.fc_out(out)  # Apply final linear layer and return output
    
# Feed Forward Neural Network
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, ff_dim),  # First linear layer
            nn.GELU(),  # Activation function
            nn.Linear(ff_dim, embed_size)  # Second linear layer
        )
        
    def forward(self, x):
        return self.net(x)  # Pass input through the feed-forward network

# Transformer Block consisting of Self-Attention and Feed Forward layers
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout):
        super().__init__()
        self.attention = SelfAttention(embed_size, num_heads)  # Self-attention layer
        self.norm1 = nn.LayerNorm(embed_size)  # Layer normalization after attention
        self.ff = FeedForward(embed_size, ff_dim)  # Feed-forward network
        self.norm2 = nn.LayerNorm(embed_size)  # Layer normalization after feed-forward network
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization
        
    def forward(self, x, mask=None):
        attn = self.attention(x, mask)  # Apply self-attention
        x = self.norm1(attn + x)  # Add residual connection and normalize
        x = self.dropout(x)  # Apply dropout
        
        ff = self.ff(x)  # Apply feed-forward network
        x = self.norm2(ff + x)  # Add residual connection and normalize
        x = self.dropout(x)  # Apply dropout
        
        return x  # Return the output of the transformer block
    
# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ff_dim, block_size, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # Token embedding layer
        self.pos_embed = nn.Embedding(block_size, embed_size)  # Positional embedding layer
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_dim, dropout)  # List of transformer blocks
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)  # Final layer normalization
        self.head = nn.Linear(embed_size, vocab_size)  # Linear layer for output logits
        self.block_size = block_size  # Maximum sequence length
        self.apply(self._init_weights)  # Initialize weights

    # Initialize weights for Linear and Embedding layers
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize linear layer weights
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # Initialize linear layer biases
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize embedding layer weights
        
    def forward(self, x, targets=None):
        B, T = x.shape  # Batch size and sequence length
        T = min(T, self.block_size)  # Ensure sequence length doesn't exceed block_size
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)  # Positional indices
        
        tok_emb = self.embed(x)  # Token embeddings
        pos_emb = self.pos_embed(pos)  # Positional embeddings
        x = tok_emb + pos_emb  # Combine token and positional embeddings
        
        for layer in self.layers:
            x = layer(x)  # Pass through each transformer block
            
        x = self.norm(x)  # Apply final layer normalization
        logits = self.head(x)  # Compute output logits
        
        if targets is None:
            loss = None  # No loss if targets are not provided
        else:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )  # Compute cross-entropy loss
            
        return logits, loss  # Return logits and loss

'''
    Key Differences from GPT-2
    Positional Encoding: Using learned embeddings instead of sinusoidal
    Normalization: LayerNorm placement differs from standard GPT
    Initialization: No pretrained weights - you control all init
    Architecture: Freedom to modify attention/FFN layers
'''
