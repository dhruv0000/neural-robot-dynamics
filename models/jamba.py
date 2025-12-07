import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Union

from models.mamba import MambaConfig, MambaBlock
from models.model_transformer import GPTConfig, Block as TransformerBlock, LayerNorm

@dataclass
class JambaConfig:
    d_model: int = 384
    n_layer: int = 6
    vocab_size: int = None
    
    # Mamba specific
    d_state: int = 16
    expand: int = 2
    dt_rank: str = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    
    # Transformer specific
    n_head: int = 12
    block_size: int = 1024
    dropout: float = 0.0
    
    # Jamba specific
    # Pattern of layers in a JambaBlock. 'm' for Mamba, 't' for Transformer.
    # Example: 'mmt' means 2 Mamba layers followed by 1 Transformer layer.
    block_pattern: str = 'mmt' 

class JambaBlock(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Create configs for sub-blocks
        self.mamba_config = MambaConfig(
            d_model=config.d_model,
            n_layer=1, # Not used in block
            d_state=config.d_state,
            expand=config.expand,
            dt_rank=config.dt_rank,
            d_conv=config.d_conv,
            conv_bias=config.conv_bias,
            bias=config.bias,
            vocab_size=config.vocab_size
        )
        
        self.gpt_config = GPTConfig(
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            n_layer=1, # Not used in block
            n_head=config.n_head,
            n_embd=config.d_model, # d_model corresponds to n_embd
            dropout=config.dropout,
            bias=config.bias
        )
        
        for layer_type in config.block_pattern:
            if layer_type == 'm':
                self.layers.append(MambaBlock(self.mamba_config))
            elif layer_type == 't':
                self.layers.append(TransformerBlock(self.gpt_config))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) # Residual connection is already inside MambaBlock? 
            # Wait, MambaBlock in mamba.py:
            # x = x + layer(x) is done in Mamba class, but MambaBlock itself returns output.
            # MambaBlock forward: returns output.
            # Transformer Block forward: x = x + attn(ln(x)); x = x + mlp(ln(x)); returns x.
            # So Transformer Block HAS residual inside.
            # MambaBlock in mamba.py: 
            # output = self.out_proj(y); return output.
            # It does NOT have residual connection added to input x inside forward.
            # But Mamba class does: x = x + layer(x).
            # So for MambaBlock, we need to add residual here.
            # For TransformerBlock, it already adds residual.
            
            # Let's check MambaBlock again.
            pass
        
        # Re-implementing forward loop to handle residuals correctly
        for i, layer in enumerate(self.layers):
            layer_type = self.config.block_pattern[i]
            if layer_type == 'm':
                 x = x + layer(x)
            elif layer_type == 't':
                 # TransformerBlock already includes residuals:
                 # x = x + self.attn(self.ln_1(x))
                 # x = x + self.mlp(self.ln_2(x))
                 # return x
                 x = layer(x)
        return x

    def forward(self, x):
        for i, layer in enumerate(self.layers):
             # MambaBlock returns the transformed features, need to add residual.
             # TransformerBlock returns x + transformed features (already residual).
             
             # To be safe, let's check the implementation of MambaBlock and Block again.
             # MambaBlock: returns output. Mamba class: x = x + layer(x).
             # Block: returns x (modified).
             
             if isinstance(layer, MambaBlock):
                 x = x + layer(x)
             else:
                 x = layer(x)
        return x

class Jamba(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()
        self.config = config
        
        if config.vocab_size is not None:
            self.embedding = nn.Linear(config.vocab_size, config.d_model)
            
        # Determine how many JambaBlocks to create
        # If n_layer is total layers, and block_pattern has length P,
        # then we have n_layer // P blocks?
        # Or n_layer is number of JambaBlocks?
        # Usually n_layer in config means "number of repeating units".
        # Let's assume n_layer is number of JambaBlocks.
        
        self.layers = nn.ModuleList([JambaBlock(config) for _ in range(config.n_layer)])
        self.norm_f = LayerNorm(config.d_model, bias=config.bias)

    def forward(self, x):
        if hasattr(self, 'embedding'):
            x = self.embedding(x)
            
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        return x
