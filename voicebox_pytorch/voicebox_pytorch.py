import math
import logging
from random import random
from functools import partial

import torch
from torch import nn, Tensor, einsum, IntTensor, FloatTensor, BoolTensor
from torch.nn import Module
import torch.nn.functional as F

import torchode as to
from torchdiffeq import odeint

from beartype import beartype
from beartype.typing import Tuple, Optional, List

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

from voicebox_pytorch.attend import Attend

from naturalspeech2_pytorch.aligner import Aligner, ForwardSumLoss, maximum_path

from audiolm_pytorch import EncodecWrapper
from spear_tts_pytorch import TextToSemantic

import torchaudio.transforms as T
from torchaudio.functional import DB_to_amplitude, resample

from vocos import Vocos

LOGGER = logging.getLogger(__file__)

# helper functions

def exists(val):
    return val is not None

def identity(t):
    return t

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def coin_flip():
    return random() < 0.5

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# mask construction helpers

def mask_from_start_end_indices(
    seq_len: int,
    start: Tensor,
    end: Tensor
):
    assert start.shape == end.shape
    device = start.device

    seq = torch.arange(seq_len, device = device, dtype = torch.long)
    seq = seq.reshape(*((-1,) * start.ndim), seq_len)
    seq = seq.expand(*start.shape, seq_len)

    mask = seq >= start[..., None].long()
    mask &= seq < end[..., None].long()
    return mask

def mask_from_frac_lengths(
    seq_len: int,
    frac_lengths: Tensor
):
    device = frac_lengths

    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.zeros_like(frac_lengths).float().uniform_(0, 1)
    start = (max_start * rand).clamp(min = 0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)

# sinusoidal positions

class LearnedSinusoidalPosEmb(Module):
    """ used by @crowsonkb """

    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered

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

class ConvPositionEmbed(Module):
    def __init__(
        self,
        dim,
        *,
        kernel_size,
        groups = None
    ):
        super().__init__()
        assert is_odd(kernel_size)
        groups = default(groups, dim) # full depthwise conv by default

        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2),
            nn.GELU()
        )

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        return rearrange(x, 'b c n -> b n c')

# norms

class RMSNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        cond_dim = None
    ):
        super().__init__()
        cond_dim = default(cond_dim, dim)
        self.scale = dim ** 0.5

        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta = nn.Linear(cond_dim, dim)

        # init to identity

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, *, cond):
        normed = F.normalize(x, dim = -1) * self.scale

        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))

        return normed * gamma + beta

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout=0,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.attend = Attend(dropout, flash = flash)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x, mask = None, rotary_emb = None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# feedforward

def FeedForward(dim, mult = 4):
    return nn.Sequential(
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
        attn_dropout=0,
        attn_flash = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None,
        use_unet_skip_connection = False,
        skip_connect_scale = None
    ):
        super().__init__()
        assert divisible_by(depth, 2)
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        if adaptive_rmsnorm:
            rmsnorm_klass = partial(AdaptiveRMSNorm, cond_dim = adaptive_rmsnorm_cond_dim_in)
        else:
            rmsnorm_klass = RMSNorm

        self.skip_connect_scale = default(skip_connect_scale, 2 ** -0.5)

        for ind in range(depth):
            layer = ind + 1
            has_skip = use_unet_skip_connection and layer > (depth // 2)

            self.layers.append(nn.ModuleList([
                nn.Linear(dim * 2, dim) if has_skip else None,
                rmsnorm_klass(dim = dim),
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=attn_flash),
                rmsnorm_klass(dim = dim),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x,
        adaptive_rmsnorm_cond = None
    ):
        skip_connects = []

        rotary_emb = self.rotary_emb(x.shape[-2])

        rmsnorm_kwargs = dict()
        if exists(adaptive_rmsnorm_cond):
            rmsnorm_kwargs = dict(cond = adaptive_rmsnorm_cond)

        for skip_combiner, attn_prenorm, attn, ff_prenorm, ff in self.layers:

            # in the paper, they use a u-net like skip connection
            # unclear how much this helps, as no ablations or further numbers given besides a brief one-two sentence mention

            if not exists(skip_combiner):
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop() * self.skip_connect_scale
                x = torch.cat((x, skip_connect), dim = -1)
                x = skip_combiner(x)

            attn_input = attn_prenorm(x, **rmsnorm_kwargs)
            x = attn(attn_input, rotary_emb = rotary_emb) + x

            ff_input = ff_prenorm(x, **rmsnorm_kwargs) 
            x = ff(ff_input) + x

        return self.final_norm(x)

# encoder decoders

class AudioEncoderDecoder(nn.Module):
    pass

class MelVoco(AudioEncoderDecoder):
    def __init__(
        self,
        *,
        log = True,
        n_mels = 100,
        sampling_rate = 24000,
        f_max = 8000,
        n_fft = 1024,
        win_length = 640,
        hop_length = 160,
        pretrained_vocos_path = 'charactr/vocos-mel-24khz'
    ):
        super().__init__()
        self.log = log
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_max = f_max
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

        self.vocos = Vocos.from_pretrained(pretrained_vocos_path)

    @property
    def latent_dim(self):
        return self.num_mels

    def encode(self, audio):
        stft_transform = T.Spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            window_fn = torch.hann_window
        )

        spectrogram = stft_transform(audio)

        mel_transform = T.MelScale(
            n_mels = self.n_mels,
            sample_rate = self.sampling_rate,
            n_stft = self.n_fft // 2 + 1,
            f_max = self.f_max
        )

        mel = mel_transform(spectrogram)

        if self.log:
            mel = T.AmplitudeToDB()(mel)

        mel = rearrange(mel, 'b d n -> b n d')
        return mel

    def decode(self, mel):
        mel = rearrange(mel, 'b n d -> b d n')

        if self.log:
            mel = DB_to_amplitude(mel, ref = 1., power = 0.5)

        return self.vocos.decode(mel)

class EncodecVoco(AudioEncoderDecoder):
    def __init__(
        self,
        *,
        sampling_rate = 24000,
        pretrained_vocos_path = 'charactr/vocos-encodec-24khz',
        bandwidth_id = 2
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encodec = EncodecWrapper()
        self.vocos = Vocos.from_pretrained(pretrained_vocos_path)

        self.register_buffer('bandwidth_id', torch.tensor([bandwidth_id]))

    @property
    def latent_dim(self):
        return self.encodec.codebook_dim

    def encode(self, audio):
        encoded_audio, _, _ = self.encodec(audio, return_encoded = True)
        return encoded_audio

    def decode(self, latents):
        _, codes, _ = self.encodec.rq(latents)
        codes = rearrange(codes, 'b n q -> b q n')

        all_audios = []
        for code in codes:
            features = self.vocos.codes_to_features(code)
            audio = self.vocos.decode(features, bandwidth_id = self.bandwidth_id)
            all_audios.append(audio)

        return torch.stack(all_audios)

# both duration and main denoising model are transformers

class DurationPredictor(Module):
    def __init__(
        self,
        *,
        num_phoneme_tokens,
        audio_enc_dec: Optional[AudioEncoderDecoder] = None,
        dim_phoneme_emb = 512,
        dim = 512,
        depth = 10,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout=0,
        attn_flash = False,
        p_drop_prob = 0.2, # p_drop in paper
        frac_lengths_mask: Tuple[float, float] = (0.1, 1.),
        aligner_kwargs: dict = dict(dim_in = 80, attn_channels = 80)
    ):
        super().__init__()

        self.audio_enc_dec = audio_enc_dec

        if exists(audio_enc_dec) and dim != audio_enc_dec.latent_dim:
            self.proj_in = nn.Linear(audio_enc_dec.latent_dim, dim)
        else:
            self.proj_in = nn.Identity()

        self.null_phoneme_id = num_phoneme_tokens # use last phoneme token as null token for CFG
        self.to_phoneme_emb = nn.Embedding(num_phoneme_tokens + 1, dim_phoneme_emb)

        self.p_drop_prob = p_drop_prob
        self.frac_lengths_mask = frac_lengths_mask

        self.to_embed = nn.Linear(dim * 2 + dim_phoneme_emb, dim)

        self.null_cond = nn.Parameter(torch.zeros(dim))

        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            attn_dropout=attn_dropout,
            attn_flash = attn_flash
        )

        self.to_pred = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...')
        )

        # aligner related

        # if we are using mel spec with 80 channels, we need to set attn_channels to 80
        # dim_in assuming we have spec with 80 channels

        self.aligner = Aligner(dim_hidden = dim_phoneme_emb, **aligner_kwargs)
        self.align_loss = ForwardSumLoss()

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    @beartype
    def forward_aligner(
        self,
        x: FloatTensor,     # (b, t, c)
        x_mask: IntTensor,  # (b, 1, t)
        y: FloatTensor,     # (b, t, c)
        y_mask: IntTensor   # (b, 1, t)
    ) -> Tuple[
        FloatTensor,        # alignment_hard: (b, t)
        FloatTensor,        # alignment_soft: (b, tx, ty)
        FloatTensor,        # alignment_logprob: (b, 1, ty, tx)
        BoolTensor          # alignment_mas: (b, tx, ty)
    ]:
        attn_mask = rearrange(x_mask, 'b 1 t -> b 1 t 1') * rearrange(y_mask, 'b 1 t -> b 1 1 t')
        alignment_soft, alignment_logprob = self.aligner(rearrange(y, 'b t c -> b c t'), x, x_mask)

        assert not torch.isnan(alignment_soft).any()

        alignment_mas = maximum_path(
            rearrange(alignment_soft, 'b 1 t1 t2 -> b t2 t1').contiguous(),
            rearrange(attn_mask, 'b 1 t1 t2 -> b t1 t2').contiguous()
        )

        alignment_hard = torch.sum(alignment_mas, -1).float()
        alignment_soft = rearrange(alignment_soft, 'b 1 t1 t2 -> b t2 t1')
        return alignment_hard, alignment_soft, alignment_logprob, alignment_mas

    def forward(
        self,
        x,
        *,
        phoneme_ids,
        mel,
        cond,
        cond_drop_prob = 0.,
        target = None,
        mask = None,
        phoneme_len = None,
        mel_len = None,
        phoneme_mask = None,
        mel_mask = None,
    ):
        batch, seq_len, cond_dim = cond.shape
        assert cond_dim == x.shape[-1]

        x, cond = map(self.proj_in, (x, cond))

        # construct mask if not given

        if not exists(mask):
            if coin_flip():
                frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask)
                mask = mask_from_frac_lengths(seq_len, frac_lengths)
            else:
                mask = prob_mask_like((batch, seq_len), self.p_drop_prob, self.device)

        cond = cond * rearrange(~mask, '... -> ... 1')

        # classifier free guidance

        if cond_drop_prob > 0.:
            cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, cond.device)

            cond = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'),
                self.null_cond,
                cond
            )

            phoneme_ids = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1'),
                self.null_phoneme_id,
                phoneme_ids
            )

        phoneme_emb = self.to_phoneme_emb(phoneme_ids)

        # aligner
        # use alignment_hard to oversample phonemes
        # Duration Predictor should predict the duration of unmasked phonemes where target is masked alignment_hard

        should_align = all([exists(el) for el in (phoneme_len, mel_len, phoneme_mask, mel_mask)])

        if should_align:
            alignment_hard, _, alignment_logprob, _ = self.forward_aligner(phoneme_emb, phoneme_mask, mel, mel_mask)
            target = alignment_hard

        # combine audio, phoneme, conditioning

        embed = torch.cat((x, phoneme_emb, cond), dim = -1)
        x = self.to_embed(embed)

        x = self.conv_embed(x) + x

        x = self.transformer(x)

        x = self.to_pred(x)

        if not exists(target):
            return x

        if not exists(mask):
            return F.l1_loss(x, target)

        loss = F.l1_loss(x, target, reduction = 'none')
        loss = loss.masked_fill(~mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den
        loss = loss.mean()
        
        if not should_align:
            return loss

        #aligner loss

        align_loss = self.align_loss(alignment_logprob, phoneme_len, mel_len)
        loss = loss + align_loss

        return loss

class VoiceBox(Module):
    def __init__(
        self,
        *,
        num_cond_tokens = None,
        audio_enc_dec: Optional[AudioEncoderDecoder] = None,
        dim_in = None,
        dim_cond_emb = 1024,
        dim = 1024,
        depth = 24,
        dim_head = 64,
        heads = 16,
        ff_mult = 4,
        time_hidden_dim = None,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout=0,
        attn_flash = False,
        p_drop_prob = 0.3, # p_drop in paper
        frac_lengths_mask: Tuple[float, float] = (0.7, 1.),
        condition_on_text = True
    ):
        super().__init__()
        dim_in = default(dim_in, dim)

        time_hidden_dim = default(time_hidden_dim, dim * 4)

        self.audio_enc_dec = audio_enc_dec

        if exists(audio_enc_dec) and dim != audio_enc_dec.latent_dim:
            self.proj_in = nn.Linear(audio_enc_dec.latent_dim, dim)
        else:
            self.proj_in = nn.Identity()

        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU()
        )

        assert not (condition_on_text and not exists(num_cond_tokens)), 'number of conditioning tokens must be specified (whether phonemes or semantic token ids) if training conditional voicebox'

        if not condition_on_text:
            dim_cond_emb = 0

        self.condition_on_text = condition_on_text
        self.num_cond_tokens = num_cond_tokens

        if condition_on_text:
            self.null_cond_id = num_cond_tokens # use last phoneme token as null token for CFG
            self.to_cond_emb = nn.Embedding(num_cond_tokens + 1, dim_cond_emb)

        self.p_drop_prob = p_drop_prob
        self.frac_lengths_mask = frac_lengths_mask

        self.to_embed = nn.Linear(dim_in * 2 + dim_cond_emb, dim)

        self.null_cond = nn.Parameter(torch.zeros(dim_in))

        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            attn_dropout=attn_dropout,
            attn_flash = attn_flash,
            adaptive_rmsnorm = True,
            adaptive_rmsnorm_cond_dim_in = time_hidden_dim
        )

        dim_out = audio_enc_dec.latent_dim if exists(audio_enc_dec) else dim_in

        self.to_pred = nn.Linear(dim, dim_out, bias = False)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        *,
        cond,
        times,
        cond_token_ids,
        cond_drop_prob = 0.1,
        target = None,
        mask = None
    ):
        batch, seq_len, cond_dim = cond.shape
        assert cond_dim == x.shape[-1]

        # project in, in case codebook dim is not equal to model dimensions

        x, cond = map(self.proj_in, (x, cond))

        # auto manage shape of times, for odeint times

        if times.ndim == 0:
            times = repeat(times, '-> b', b = cond.shape[0])

        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = cond.shape[0])

        # construct mask if not given

        if not exists(mask):
            if coin_flip():
                frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask)
                mask = mask_from_frac_lengths(seq_len, frac_lengths)
            else:
                mask = prob_mask_like((batch, seq_len), self.p_drop_prob, self.device)

        cond = cond * rearrange(~mask, '... -> ... 1')

        # classifier free guidance

        if cond_drop_prob > 0.:
            cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, self.device)

            cond = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'),
                self.null_cond,
                cond
            )

            cond_ids = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1'),
                self.null_cond_id,
                cond_token_ids
            )

        # phoneme or semantic conditioning embedding

        cond_emb = None

        if self.condition_on_text:
            cond_emb = self.to_cond_emb(cond_token_ids)

            cond_emb_length = cond_emb.shape[-2]
            if cond_emb_length != seq_len:
                cond_emb = rearrange(cond_emb, 'b n d -> b d n 1')
                cond_emb = F.interpolate(cond_emb, (seq_len, 1), mode = 'bilinear')
                cond_emb = rearrange(cond_emb, 'b d n 1 -> b n d')

        # concat source signal, semantic / phoneme conditioning embed, and conditioning
        # and project

        to_concat = [*filter(exists, (x, cond_emb, cond))]
        embed = torch.cat(to_concat, dim = -1)

        x = self.to_embed(embed)

        x = self.conv_embed(x) + x

        time_emb = self.sinu_pos_emb(times)

        # attend

        x = self.transformer(x, adaptive_rmsnorm_cond = time_emb)

        x = self.to_pred(x)

        # if no target passed in, just return logits

        if not exists(target):
            return x

        if not exists(mask):
            return F.mse_loss(x, target)

        loss = F.mse_loss(x, target, reduction = 'none')

        loss = reduce(loss, 'b n d -> b n', 'mean')
        loss = loss.masked_fill(~mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den

        return loss.mean()

# wrapper for the CNF

def is_probably_audio_from_shape(t):
    return t.ndim == 2 or (t.ndim == 3 and t.shape[1] == 1)

class ConditionalFlowMatcherWrapper(Module):
    @beartype
    def __init__(
        self,
        voicebox: VoiceBox,
        text_to_semantic: Optional[TextToSemantic] = None,
        sigma = 0.,
        ode_atol = 1e-5,
        ode_rtol = 1e-5,
        ode_step_size = 0.0625,
        use_torchode = False,
        torchdiffeq_ode_method = 'midpoint',   # use midpoint for torchdiffeq, as in paper
        torchode_method_klass = to.Tsit5,      # use tsit5 for torchode, as torchode does not have midpoint (recommended by Bryan @b-chiang)
        cond_drop_prob = 0.
    ):
        super().__init__()
        self.sigma = sigma

        self.voicebox = voicebox
        self.condition_on_text = voicebox.condition_on_text
        self.text_to_semantic = text_to_semantic

        assert not (not self.condition_on_text and exists(text_to_semantic)), 'TextToSemantic should not be passed in if not conditioning on text'
        assert not (exists(text_to_semantic) and not exists(text_to_semantic.wav2vec)), 'the wav2vec module must exist on the TextToSemantic, if being used to condition on text'

        self.cond_drop_prob = cond_drop_prob

        self.use_torchode = use_torchode
        self.torchode_method_klass = torchode_method_klass

        self.odeint_kwargs = dict(
            atol = ode_atol,
            rtol = ode_rtol,
            method = torchdiffeq_ode_method,
            options = dict(step_size = ode_step_size)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def sample(
        self,
        *,
        cond,
        texts: Optional[List[str]] = None,
        text_token_ids: Optional[Tensor] = None,
        semantic_token_ids = None,
        phoneme_ids = None,
        mask = None,
        steps = 3,
        cond_scale = 1.,
        decode_to_audio = True
    ):
        shape = cond.shape
        batch = shape[0]

        # take care of condition as raw audio

        cond_is_raw_audio = is_probably_audio_from_shape(cond)

        if cond_is_raw_audio:
            assert exists(self.voicebox.audio_enc_dec)

            self.voicebox.audio_enc_dec.eval()
            cond = self.voicebox.audio_enc_dec.encode(cond)

        # setup text conditioning, either coming from duration model (as phoneme ids)
        # for coming from text-to-semantic module from spear-tts paper, as (semantic ids)

        num_cond_inputs = sum([*map(exists, (texts, text_token_ids, semantic_token_ids, phoneme_ids))])
        assert num_cond_inputs <= 1

        mask = None
        cond_token_ids = None

        if self.condition_on_text:
            if exists(self.text_to_semantic):
                assert not exists(phoneme_ids)

                if not exists(semantic_token_ids):

                    semantic_token_ids, mask = self.text_to_semantic.generate(
                        source = default(text_token_ids, texts),
                        source_type = 'text',
                        target_type = 'speech',
                        max_length = 10,
                        return_target_mask = True
                    )

                cond_token_ids = semantic_token_ids
            else:
                cond_token_ids = phoneme_ids

            cond_length = cond.shape[-2]
            cond_tokens_seq_len = cond_token_ids.shape[-1]

            # curtail or pad (todo: generalize this to any dimension and put in a function)

            if cond_length > cond_tokens_seq_len:
                cond = cond[:, :cond_tokens_seq_len, :]
            elif cond_length < cond_tokens_seq_len:
                cond = F.pad(cond, (0, 0, 0, cond_tokens_seq_len), value = 0.)

        else:
            assert num_cond_inputs == 0, 'no conditioning inputs should be given if not conditioning on text'

        # neural ode

        self.voicebox.eval()

        def fn(t, x, *, packed_shape = None):
            if exists(packed_shape):
                x = unpack_one(x, packed_shape, 'b *')

            out = self.voicebox.forward_with_cond_scale(
                x,
                times = t,
                cond_token_ids = cond_token_ids,
                cond = cond,
                cond_scale = cond_scale,
                mask = mask
            )

            if exists(packed_shape):
                out = rearrange(out, 'b ... -> b (...)')

            return out

        y0 = torch.randn_like(cond)
        t = torch.linspace(0, 1, steps, device = self.device)

        if not self.use_torchode:
            LOGGER.debug('sampling with torchdiffeq')

            trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
            sampled = trajectory[-1]
        else:
            LOGGER.debug('sampling with torchode')

            t = repeat(t, 'n -> b n', b = batch)
            y0, packed_shape = pack_one(y0, 'b *')

            fn = partial(fn, packed_shape = packed_shape)

            term = to.ODETerm(fn)
            step_method = self.torchode_method_klass(term = term)

            step_size_controller = to.IntegralController(
                atol = self.odeint_kwargs['atol'],
                rtol = self.odeint_kwargs['rtol'],
                term = term
            )

            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            jit_solver = torch.compile(solver)

            init_value = to.InitialValueProblem(y0 = y0, t_eval = t)

            sol = jit_solver.solve(init_value)

            sampled = sol.ys[:, -1]
            sampled = unpack_one(sampled, packed_shape, 'b *')

        if not decode_to_audio or not exists(self.voicebox.audio_enc_dec):
            return sampled

        return self.voicebox.audio_enc_dec.decode(sampled)

    def forward(
        self,
        x1,
        *,
        cond,
        semantic_token_ids = None,
        phoneme_ids = None,
        mask = None,
        input_sampling_rate = None # will assume it to be the same as the audio encoder decoder sampling rate, if not given. if given, will resample
    ):
        """
        following eq (5) (6) in https://arxiv.org/pdf/2306.15687.pdf
        """

        batch, seq_len, dtype, σ = *x1.shape[:2], x1.dtype, self.sigma

        # if raw audio is given, convert if audio encoder / decoder was passed in

        input_is_raw_audio, cond_is_raw_audio = map(is_probably_audio_from_shape, (x1, cond))

        if input_is_raw_audio:
            raw_audio = x1

        if any([input_is_raw_audio, cond_is_raw_audio]):
            assert exists(self.voicebox.audio_enc_dec), 'audio_enc_dec must be set on VoiceBox to train directly on raw audio'

            audio_enc_dec_sampling_rate = self.voicebox.audio_enc_dec.sampling_rate
            input_sampling_rate = default(input_sampling_rate, audio_enc_dec_sampling_rate)

            with torch.no_grad():
                self.voicebox.audio_enc_dec.eval()

                if input_is_raw_audio:
                    x1 = resample(x1, input_sampling_rate, audio_enc_dec_sampling_rate)
                    x1 = self.voicebox.audio_enc_dec.encode(x1)

                if cond_is_raw_audio:
                    cond = resample(cond, input_sampling_rate, audio_enc_dec_sampling_rate)
                    cond = self.voicebox.audio_enc_dec.encode(cond)

        # setup text conditioning, either coming from duration model (as phoneme ids)
        # or from text-to-semantic module, semantic ids encoded with wav2vec (hubert usually)

        assert self.condition_on_text or not (exists(semantic_token_ids) or exists(phoneme_ids)), 'semantic or phoneme ids should not be passed in if not conditioning on text'

        cond_token_ids = None

        if self.condition_on_text:
            if exists(self.text_to_semantic):
                assert not exists(phoneme_ids), 'phoneme ids are not needed for conditioning with spear-tts text-to-semantic'

                if not exists(semantic_token_ids):
                    assert input_is_raw_audio
                    wav2vec = self.text_to_semantic.wav2vec
                    wav2vec_input = resample(raw_audio, input_sampling_rate, wav2vec.target_sample_hz)
                    semantic_token_ids = wav2vec(wav2vec_input).clone()

                cond_token_ids = semantic_token_ids
            else:
                cond_token_ids = phoneme_ids

        # main conditional flow logic is below

        # x0 is gaussian noise

        x0 = torch.randn_like(x1)

        # random times

        times = torch.rand((batch,), dtype = dtype, device = self.device)
        t = rearrange(times, 'b -> b 1 1')

        # sample xt (w in the paper)

        w = (1 - (1 - σ) * t) * x0 + t * x1

        flow = x1 - (1 - σ) * x0

        # predict

        self.voicebox.train()

        loss = self.voicebox(
            w,
            cond = cond,
            mask = mask,
            times = times,
            target = flow,
            cond_token_ids = cond_token_ids,
            cond_drop_prob = self.cond_drop_prob
        )

        return loss
