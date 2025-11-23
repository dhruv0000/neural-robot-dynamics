import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class MambaConfig:
    d_model: int = 384
    n_layer: int = 6
    d_state: int = 16
    expand: int = 2
    dt_rank: str = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    vocab_size: int = None

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if config.dt_rank == 'auto' else config.dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def forward(self, x):
        # x: (B, L, D)
        batch, seq_len, d_model = x.shape
        
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        (x_in, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :seq_len]
        x_in = x_in.transpose(1, 2)

        x_in = F.silu(x_in)

        ssm_params = self.x_proj(x_in)
        dt, B, C = torch.split(ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        y = self.selective_scan(x_in, dt, self.A_log, B, C, self.D)
        
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def selective_scan(self, u, dt, A_log, B, C, D):
        # u: (B, L, d_inner)
        # dt: (B, L, d_inner)
        # A_log: (d_inner, d_state)
        # B: (B, L, d_state)
        # C: (B, L, d_state)
        # D: (d_inner)
        
        batch_size, seq_len, d_inner = u.shape
        d_state = A_log.shape[1]

        A = -torch.exp(A_log)  # (d_inner, d_state)
        
        # Discretize A and B
        # dt: (B, L, d_inner) -> (B, L, d_inner, 1)
        dt_expanded = dt.unsqueeze(-1)
        
        # dA: (B, L, d_inner, d_state)
        dA = torch.exp(dt_expanded * A) 
        
        # dB: (B, L, d_inner, d_state)
        dB = dt_expanded * B.unsqueeze(2)

        # Scan
        h = torch.zeros(batch_size, d_inner, d_state, device=u.device)
        ys = []
        
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1) # (B, d_inner)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # (B, L, d_inner)
        y = y + u * D
        return y

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        if config.vocab_size is not None:
            self.embedding = nn.Linear(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.d_model)

    def forward(self, x):
        # x: (B, L, D) or (B, L, vocab_size)
        if hasattr(self, 'embedding'):
            x = self.embedding(x)
            
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm_f(x)
        return x
