import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ====================================================
#                 Causal Self-Attention
# ====================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len=2048):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        # masked causal attention buffer
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(out)

# ====================================================
#                       Mamba Layer
# ====================================================
class MambaLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(
            in_channels=d_model * 2,
            out_channels=d_model * 2,
            kernel_size=4,
            padding=3,
            groups=d_model * 2  # depthwise
        )
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        B, T, C = x.shape
        x_proj = self.in_proj(x)

        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv(x_conv)[:, :, :T]   # keep exact T length
        x_conv = x_conv.transpose(1, 2)

        return self.out_proj(self.act(x_conv))


# ====================================================
#                        MLP
# ====================================================
class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ====================================================
#                     Jamba Block
# ====================================================
class JambaBlock(nn.Module):
    def __init__(self, d_model, use_attention=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mixer = (
            CausalSelfAttention(d_model, n_head=4)
            if use_attention else
            MambaLayer(d_model)
        )
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ====================================================
#                    Jamba NeRD Model
# ====================================================
class JambaModel(nn.Module):
    def __init__(self, input_dim, d_model=256, n_layers=16, vocab_size=None):
        super().__init__()

        self.embed = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            use_attn = ((i + 1) % 8 == 0)  # every 8th layer uses attention
            self.layers.append(JambaBlock(d_model, use_attention=use_attn))

        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, input_dim, bias=False)

        # For NeRD trainer compatibility
        self.normalize_output = False
        self.input_rms = None
        self.output_rms = None
        self.is_rnn = False

    # ---------------- NeRD-required methods ----------------
    def set_input_rms(self, rms):
        self.input_rms = rms

    def set_output_rms(self, rms):
        self.output_rms = rms

    def init_rnn(self, batch_size):
        pass

    def evaluate(self, x):
        return self.forward(x)
    # --------------------------------------------------------

    def forward(self, x):
        # handle dictionary input
        if isinstance(x, dict):
            x = x.get("net_input", x.get("states", list(x.values())[0]))

        x = self.embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        return self.head(x)
