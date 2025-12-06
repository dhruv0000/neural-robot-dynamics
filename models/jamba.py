import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Components ---

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len=2048):
        super().__init__()
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.d_model = d_model
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MambaLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        x_proj = self.in_proj(x)
        x_conv = x_proj.transpose(1, 2) 
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        return self.out_proj(self.act(x_conv))

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    def forward(self, x): return self.net(x)

class JambaBlock(nn.Module):
    def __init__(self, d_model, layer_idx, use_attention=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if use_attention:
            self.mixer = CausalSelfAttention(d_model, n_head=4)
        else:
            self.mixer = MambaLayer(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# --- Main Model ---

class JambaModel(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, vocab_size=None):
        """
        input_dim: The number of features in your robot state (e.g., 28, 64, etc.)
        """
        super().__init__()
        # PHYSICS FIX: Use Linear instead of Embedding
        self.embed = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([])
        
        attn_interval = 8
        for i in range(n_layers):
            is_attn = ((i + 1) % attn_interval == 0)
            self.layers.append(JambaBlock(d_model, i, is_attn))
            
        self.norm_f = nn.LayerNorm(d_model)
        # We output the same dimension as input to predict the NEXT state
        self.head = nn.Linear(d_model, input_dim, bias=False)

    def forward(self, x):
        # HANDLE DICTIONARY INPUT
        if isinstance(x, dict):
            # If the data loader provides a dict, we try to find the inputs
            if 'net_input' in x:
                x = x['net_input']
            elif 'states' in x:
                # Fallback: just use states (concatenate actions if available)
                x = x['states']
            else:
                raise ValueError("JambaModel received a dict but couldn't find 'net_input' or 'states'")

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.head(x)