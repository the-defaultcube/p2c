import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Injects positional information into token embeddings."""
    
    def __init__(self, d_model: int = 512, max_len: int = 5000):
        """
        Initialize positional encoding layer.
        
        Args:
            d_model: Dimension of the token embeddings.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to the input tensor.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            Tensor with added positional encodings.
        """
        return x + self.pe[:x.size(1)]

class MultiHeadAttention(nn.Module):
    """Implements multi-head scaled dot-product attention."""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        """
        Initialize multi-head attention layer.
        
        Args:
            d_model: Dimension of the input embeddings.
            n_heads: Number of attention heads.
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * self.d_v, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_v, d_model, bias=False)
        
        self._reset_parameters()
        
    def _reset_parameters(self) -> None:
        """Initialize parameters with Xavier normal initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=1.0)
        
    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the multi-head attention layer.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, d_model).
            k: Key tensor of shape (batch_size, seq_len, d_model).
            v: Value tensor of shape (batch_size, seq_len, d_model).
            mask: Attention mask (optional).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size = q.size(0)
        seq_len = q.size(1)
        
        # Project queries, keys, and values
        q = self.W_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        
        # Compute output
        output = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_v)
        output = self.W_o(output)
        
        return output

class EncoderLayer(nn.Module):
    """Implements a single encoder layer with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, n_heads: int = 8, dropout: float = 0.1):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Dimension of the input embeddings.
            d_ff: Dimension of the feed-forward network.
            n_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-attention
        attn_output = self.self_attn(x, x, x)
        
        # Residual connection and normalization
        x = self.norm(x + self.dropout(attn_output))
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        
        # Residual connection and normalization
        x = self.norm(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """Implements a single decoder layer with self-attention, encoder-decoder attention, and feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, n_heads: int = 8, dropout: float = 0.1):
        """
        Initialize decoder layer.
        
        Args:
            d_model: Dimension of the input embeddings.
            d_ff: Dimension of the feed-forward network.
            n_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the decoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            memory: Output of the encoder of shape (batch_size, seq_len, d_model).
            src_mask: Source attention mask (optional).
            tgt_mask: Target attention mask (optional).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-attention
        x = self.norm(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        
        # Encoder-decoder attention
        x = self.norm(x + self.dropout(self.enc_dec_attn(x, memory, memory, src_mask)))
        
        # Feed-forward network
        x = self.norm(x + self.dropout(self.feed_forward(x)))
        
        return x

class Model(nn.Module):
    """Implements the Transformer model."""
    
    def __init__(self, 
                 d_model: int = 512, 
                 n_heads: int = 8, 
                 num_layers: int = 6, 
                 dropout: float = 0.1):
        """
        Initialize the Transformer model.
        
        Args:
            d_model: Dimension of the input embeddings.
            n_heads: Number of attention heads.
            num_layers: Number of layers in encoder and decoder.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize encoder and decoder
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, 2048, n_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, 2048, n_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Final linear projection
        self.linear = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Transformer model.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len).
            tgt: Target tensor of shape (batch_size, tgt_seq_len).
            src_mask: Source attention mask (optional).
            tgt_mask: Target attention mask (optional).
            
        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size).
        """
        # Embeddings and positional encoding
        src_emb = self.positional_encoding(src)
        tgt_emb = self.positional_encoding(tgt)
        
        # Encoder
        for layer in self.encoder:
            src_emb = layer(src_emb)
        
        # Decoder
        for layer in self.decoder:
            tgt_emb = layer(tgt_emb, src_emb, src_mask, tgt_mask)
        
        # Final projection
        output = self.linear(tgt_emb)
        
        return output

# Example usage
if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Transformer Model')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to the configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = Model(
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    # Example input
    batch_size = 32
    src_seq_len = 50
    tgt_seq_len = 50
    src = torch.randint(0, config['dataset']['vocab_size'], (batch_size, src_seq_len))
    tgt = torch.randint(0, config['dataset']['vocab_size'], (batch_size, tgt_seq_len))
    
    # Create masks
    src_mask = torch.ones(batch_size, 1, src_seq_len, src_seq_len)
    tgt_mask = torch.triu(torch.ones(batch_size, 1, tgt_seq_len, tgt_seq_len), diagonal=1).bool()
    
    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)
    
    logger.info(f"Output shape: {output.shape}")
