"""
Shared model architectures for paralinguistic processing.

This module contains the definitive implementations of neural network
architectures used across the paralinguistic pipeline to avoid duplication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with 1D convolutions for acoustic feature processing."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class UpgradedAcousticEncoder(nn.Module):
    """
    Upgraded acoustic encoder with CNN + Transformer architecture.
    
    This is the definitive implementation used across the paralinguistic pipeline.
    It takes eGeMAPS features and produces normalized embeddings.
    
    Architecture:
    1. CNN feature extractor with residual blocks
    2. Transformer encoder with CLS token
    3. Projection head with normalization
    
    Args:
        n_channels: Number of input acoustic features (default: 88 for eGeMAPS)
        embedding_dim: Output embedding dimension (default: 512)
        nhead: Number of attention heads in transformer (default: 16)
        num_encoder_layers: Number of transformer layers (default: 8)
        dim_feedforward: Transformer feedforward dimension (default: 2048)
    """
    
    def __init__(self, n_channels=88, embedding_dim=512, nhead=16, num_encoder_layers=8, dim_feedforward=2048):
        super().__init__()
        
        # CNN feature extractor with residual blocks
        self.cnn_extractor = nn.Sequential(
            ResidualBlock(n_channels, 64),
            nn.Dropout(0.3),
            ResidualBlock(64, 128),
            nn.Dropout(0.3),
            ResidualBlock(128, 256),
            nn.Dropout(0.3),
            ResidualBlock(256, 512),
            nn.Dropout(0.3),
            ResidualBlock(512, 512)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=0.3, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # CLS token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x, mask=None):
        """
        Forward pass through the acoustic encoder.
        
        Args:
            x: Input tensor of shape [batch, n_channels, seq_len]
            mask: Optional mask tensor of shape [batch, seq_len]
                  True for valid positions, False for padding
        
        Returns:
            Normalized embedding tensor of shape [batch, embedding_dim]
        """
        # CNN feature extraction
        # x: [batch, 88, seq_len] -> [batch, 512, seq_len]
        x = self.cnn_extractor(x)
        
        # Transpose for transformer: [batch, 512, seq_len] -> [batch, seq_len, 512]
        x = x.permute(0, 2, 1)
        
        # Add CLS token at the beginning
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # -> [batch, seq_len + 1, 512]
        
        # Prepare attention mask for transformer
        if mask is not None:
            # Add mask for CLS token (always attended)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            # Transformer expects True for positions to ignore
            mask = ~mask

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Extract CLS token representation
        cls_output = x[:, 0]  # [batch, 512]
        
        # Project to final embedding dimension
        embedding = self.projection(cls_output)
        
        # L2 normalize the embedding
        normalized = F.normalize(embedding, dim=1)
        
        return normalized
