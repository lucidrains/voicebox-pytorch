import torch
from torch import nn, Tensor, einsum
from torch.nn import Module
import torch.nn.functional as F

from beartype import beartype

from einops import rearrange, repeat, reduce

from voicebox_pytorch.attend import Attend

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

# rotary positional embeddings
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# convolutional positional generating module

def DepthWiseConv1d(
    dim,
    kernel_size
):
    assert not divisible_by(kernel_size, 2)
    return nn.Conv1d(dim, dim, kernel_size, groups = dim, padding = kernel_size // 2)

class ConvPositionEmbed(Module):
    def __init__(
        self,
        dim,
        *,
        kernel_size
    ):
        super().__init__()
        self.dw_conv1d = nn.Sequential(
            DepthWiseConv1d(dim, kernel_size),
            nn.GELU()
        )

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        return rearrange(x, 'b c n -> b n c')

# norms

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.attend = Attend(flash = flash)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x, mask = None, rotary_emb = None):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(t, rotary_emb), (q, k))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# feedforward

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )

# transformer

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_flash = False
    ):
        super().__init__()
        assert divisible_by(depth, 2)

        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        for ind in range(depth):
            layer = ind + 1
            has_skip = layer > (depth // 2)

            self.layers.append(nn.ModuleList([
                nn.Linear(dim * 2, dim) if has_skip else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, flash = attn_flash),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x):
        skip_connects = []

        rotary_emb = self.rotary_emb(x.shape[-2])

        for skip_combiner, attn, ff in self.layers:

            # in the paper, they use a u-net like skip connection
            # unclear how much this helps, as no ablations or further numbers given besides a brief one-two sentence mention

            if not exists(skip_combiner):
                skip_connects.append(x)
            else:
                x = torch.cat((x, skip_connects.pop()), dim = -1)
                x = skip_combiner(x)

            x = attn(x, rotary_emb = rotary_emb) + x
            x = ff(x) + x

        return x

# both duration and main denoising model are transformers

class DurationPredictor(Module):
    def __init__(
        self,
        dim = 512,
        *,
        depth = 10,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_pos_embed_kernel_size = 31,
        attn_flash = False
    ):
        super().__init__()
        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            attn_flash = attn_flash
        )

    def forward(self, x):
        x = self.conv_embed(x) + x
        x = self.transformer(x)
        return x

class VoiceBox(Module):
    def __init__(
        self,
        dim = 1024,
        *,
        depth = 24,
        dim_head = 64,
        heads = 16,
        ff_mult = 4,
        conv_pos_embed_kernel_size = 31,
        attn_flash = False
    ):
        super().__init__()
        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            attn_flash = attn_flash
        )

    def forward(self, x):
        x = self.conv_embed(x) + x
        x = self.transformer(x)
        return x

# wrapper for the CNF

class CNFWrapper(Module):
    @beartype
    def __init__(
        self,
        voicebox: VoiceBox
    ):
        super().__init__()

    def forward(self, x):
        return x
