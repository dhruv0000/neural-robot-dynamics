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
        # 1. Expansion: Project input up to 2x dimensions
        self.in_proj = nn.Linear(d_model, d_model * 2)
        
        # 2. Convolution: Must handle the expanded size (d_model * 2)
        self.conv1d = nn.Conv1d(
            in_channels=d_model * 2,   # <--- FIXED: 256 inputs
            out_channels=d_model * 2,  # <--- FIXED: 256 outputs
            kernel_size=4, 
            groups=d_model * 2,        # <--- FIXED: Depthwise over 256
            padding=3
        )
        
        # 3. Projection: Project back down to d_model
        self.out_proj = nn.Linear(d_model * 2, d_model) # <--- FIXED
        
        self.act = nn.SiLU() # <--- FIXED TYPO

    def forward(self, x):
        # x: [Batch, Len, Dim]
        B, L, D = x.shape
        
        # Project up
        x_proj = self.in_proj(x)
        
        # Transpose for Conv1d: [Batch, Dim, Len]
        x_conv = x_proj.transpose(1, 2) 
        
        # Convolve (truncate to length L because padding adds extra)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        
        # Transpose back: [Batch, Len, Dim]
        x_conv = x_conv.transpose(1, 2)
        
        # Activation and project down
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
        self.embed = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([])
        
        # Helper variables for Trainer
        self.normalize_output = False 
        self.input_rms = None
        self.output_rms = None
        self.is_rnn = False # Jamba acts like a Transformer here, not an RNN
        
        attn_interval = 8
        for i in range(n_layers):
            is_attn = ((i + 1) % attn_interval == 0)
            self.layers.append(JambaBlock(d_model, i, is_attn))
            
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, input_dim, bias=False)

    # --- HELPER METHODS REQUIRED BY TRAINER ---
    def set_input_rms(self, input_rms):
        self.input_rms = input_rms

    def set_output_rms(self, output_rms):
        self.output_rms = output_rms
        
    def init_rnn(self, batch_size):
        # We need this method because 'evaluator.py' calls it blindly.
        # Since Jamba isn't an RNN, we just pass.
        pass
    # ------------------------------------------

    def forward(self, x):
        # HANDLE DICTIONARY INPUT
        if isinstance(x, dict):
            if 'net_input' in x:
                x = x['net_input']
            elif 'states' in x:
                x = x['states']
            else:
                x = list(x.values())[0]

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.head(x)