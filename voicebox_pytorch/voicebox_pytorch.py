import math
import logging
from random import random
from functools import partial
from pathlib import Path

import torch
from torch import nn, Tensor, einsum, IntTensor, FloatTensor, BoolTensor
from torch.nn import Module
import torch.nn.functional as F

import torchode as to
from torchdiffeq import odeint

from beartype import beartype
from beartype.typing import Tuple, Optional, List, Union

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

from voicebox_pytorch.attend import Attend

from naturalspeech2_pytorch.aligner import Aligner, ForwardSumLoss, BinLoss, maximum_path
from naturalspeech2_pytorch.utils.tokenizer import Tokenizer
from naturalspeech2_pytorch.naturalspeech2_pytorch import generate_mask_from_repeats

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

def reduce_masks_with_and(*masks):
    masks = [*filter(exists, masks)]

    if len(masks) == 0:
        return None

    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask

def interpolate_1d(t, length, mode = 'bilinear'):
    " pytorch does not offer interpolation 1d, so hack by converting to 2d "

    dtype = t.dtype
    t = t.float()

    implicit_one_channel = t.ndim == 2
    if implicit_one_channel:
        t = rearrange(t, 'b n -> b 1 n')

    t = rearrange(t, 'b d n -> b d n 1')
    t = F.interpolate(t, (length, 1), mode = mode)
    t = rearrange(t, 'b d n 1 -> b d n')

    if implicit_one_channel:
        t = rearrange(t, 'b 1 n -> b n')

    t = t.to(dtype)
    return t

def curtail_or_pad(t, target_length):
    length = t.shape[-2]

    if length > target_length:
        t = t[..., :target_length, :]
    elif length < target_length:
        t = F.pad(t, (0, 0, 0, target_length - length), value = 0.)

    return t

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
    def __init__(self, dim, theta = 50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @beartype
    def forward(self, t: Union[int, Tensor]):
        if not torch.is_tensor(t):
            t = torch.arange(t, device = self.device)

        t = t.type_as(self.inv_freq)
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

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0,
        flash = False,
        qk_norm = False,
        qk_norm_scale = 10
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads

        scale = qk_norm_scale if qk_norm else None

        self.attend = Attend(dropout, flash = flash, scale = scale)

        self.qk_norm = qk_norm

        if qk_norm:
            self.q_norm = MultiheadRMSNorm(dim_head, heads = heads)
            self.k_norm = MultiheadRMSNorm(dim_head, heads = heads)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x, mask = None, rotary_emb = None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
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
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_register_tokens = 0.,
        attn_flash = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None,
        use_unet_skip_connection = False,
        skip_connect_scale = None,
        attn_qk_norm = False
    ):
        super().__init__()
        assert divisible_by(depth, 2)
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0

        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

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
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash, qk_norm = attn_qk_norm),
                rmsnorm_klass(dim = dim),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.final_norm = RMSNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        mask = None,
        adaptive_rmsnorm_cond = None
    ):
        batch, seq_len, *_ = x.shape

        # add register tokens to the left

        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)

            x, ps = pack([register_tokens, x], 'b * d')

            if exists(mask):
                mask = F.pad(mask, (self.num_register_tokens, 0), value = True)

        # keep track of skip connections

        skip_connects = []

        # rotary embeddings

        positions = seq_len

        if self.has_register_tokens:
            main_positions = torch.arange(seq_len, device = self.device, dtype = torch.long)
            register_positions = torch.full((self.num_register_tokens,), -10000, device = self.device, dtype = torch.long)
            positions = torch.cat((register_positions, main_positions))

        rotary_emb = self.rotary_emb(positions)

        # adaptive rmsnorm

        rmsnorm_kwargs = dict()
        if exists(adaptive_rmsnorm_cond):
            rmsnorm_kwargs = dict(cond = adaptive_rmsnorm_cond)

        # going through the attention layers

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
            x = attn(attn_input, mask = mask, rotary_emb = rotary_emb) + x

            ff_input = ff_prenorm(x, **rmsnorm_kwargs) 
            x = ff(ff_input) + x

        # remove the register tokens

        if self.has_register_tokens:
            _, x = unpack(x, ps, 'b * d')

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
    def downsample_factor(self):
        raise NotImplementedError

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
    def downsample_factor(self):
        return self.encodec.downsample_factor

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
    @beartype
    def __init__(
        self,
        *,
        audio_enc_dec: Optional[AudioEncoderDecoder] = None,
        tokenizer: Optional[Tokenizer] = None,
        num_phoneme_tokens: Optional[int] = None,
        dim_phoneme_emb = 512,
        dim = 512,
        depth = 10,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_qk_norm = True,
        ff_dropout = 0.,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout=0,
        attn_flash = False,
        p_drop_prob = 0.2, # p_drop in paper
        frac_lengths_mask: Tuple[float, float] = (0.1, 1.),
        aligner_kwargs: dict = dict(dim_in = 80, attn_channels = 80)
    ):
        super().__init__()

        # audio encoder / decoder

        self.audio_enc_dec = audio_enc_dec

        if exists(audio_enc_dec) and dim != audio_enc_dec.latent_dim:
            self.proj_in = nn.Linear(audio_enc_dec.latent_dim, dim)
        else:
            self.proj_in = nn.Identity()

        # phoneme related

        assert not (exists(tokenizer) and exists(num_phoneme_tokens)), 'if a phoneme tokenizer was passed into duration module, number of phoneme tokens does not need to be specified'

        if not exists(tokenizer) and not exists(num_phoneme_tokens):
            tokenizer = Tokenizer() # default to english phonemes with espeak

        if exists(tokenizer):
            num_phoneme_tokens = tokenizer.vocab_size

        self.tokenizer = tokenizer

        self.to_phoneme_emb = nn.Embedding(num_phoneme_tokens, dim_phoneme_emb)

        self.p_drop_prob = p_drop_prob
        self.frac_lengths_mask = frac_lengths_mask

        self.to_embed = nn.Linear(dim + dim_phoneme_emb, dim)

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
            ff_dropout = ff_dropout,
            attn_dropout=attn_dropout,
            attn_flash = attn_flash,
            attn_qk_norm = attn_qk_norm
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

    def align_phoneme_ids_with_durations(self, phoneme_ids, durations):
        repeat_mask = generate_mask_from_repeats(durations.clamp(min = 1))
        aligned_phoneme_ids = einsum('b i, b i j -> b j', phoneme_ids.float(), repeat_mask.float()).long()
        return aligned_phoneme_ids

    @torch.inference_mode()
    @beartype
    def forward_with_cond_scale(
        self,
        *args,
        texts: Optional[List[str]] = None,
        phoneme_ids = None,
        cond_scale = 1.,
        return_aligned_phoneme_ids = False,
        **kwargs
    ):
        if exists(texts):
            phoneme_ids = self.tokenizer.texts_to_tensor_ids(texts)

        forward_kwargs = dict(
            return_aligned_phoneme_ids = False,
            phoneme_ids = phoneme_ids
        )

        durations = self.forward(*args, cond_drop_prob = 0., **forward_kwargs, **kwargs)

        if cond_scale == 1.:
            if not return_aligned_phoneme_ids:
                return durations

            return durations, self.align_phoneme_ids_with_durations(phoneme_ids, durations)

        null_durations = self.forward(*args, cond_drop_prob = 1., **forward_kwargs, **kwargs)
        scaled_durations = null_durations + (durations - null_durations) * cond_scale

        if not return_aligned_phoneme_ids:
            return scaled_durations

        return scaled_durations, self.align_phoneme_ids_with_durations(phoneme_ids, scaled_durations)

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

    @beartype
    def forward(
        self,
        *,
        cond,
        texts: Optional[List[str]] = None,
        phoneme_ids = None,
        cond_drop_prob = 0.,
        target = None,
        cond_mask = None,
        mel = None,
        phoneme_len = None,
        mel_len = None,
        phoneme_mask = None,
        mel_mask = None,
        self_attn_mask = None,
        return_aligned_phoneme_ids = False
    ):
        batch, seq_len, cond_dim = cond.shape

        cond = self.proj_in(cond)

        # text to phonemes, if tokenizer is given

        if not exists(phoneme_ids):
            assert exists(self.tokenizer)
            phoneme_ids = self.tokenizer.texts_to_tensor_ids(texts)

        # construct mask if not given

        if not exists(cond_mask):
            if coin_flip():
                frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask)
                cond_mask = mask_from_frac_lengths(seq_len, frac_lengths)
            else:
                cond_mask = prob_mask_like((batch, seq_len), self.p_drop_prob, self.device)

        cond = cond * rearrange(~cond_mask, '... -> ... 1')

        # classifier free guidance

        if cond_drop_prob > 0.:
            cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, cond.device)

            cond = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'),
                self.null_cond,
                cond
            )

        # phoneme id of -1 is padding

        if not exists(self_attn_mask):
            self_attn_mask = phoneme_ids != -1

        phoneme_ids = phoneme_ids.clamp(min = 0)

        # get phoneme embeddings

        phoneme_emb = self.to_phoneme_emb(phoneme_ids)

        # force condition to be same length as input phonemes

        cond = curtail_or_pad(cond, phoneme_ids.shape[-1])

        # combine audio, phoneme, conditioning

        embed = torch.cat((phoneme_emb, cond), dim = -1)
        x = self.to_embed(embed)

        x = self.conv_embed(x) + x

        x = self.transformer(
            x,
            mask = self_attn_mask
        )

        durations = self.to_pred(x)

        if not self.training:
            if not return_aligned_phoneme_ids:
                return durations

            return durations, self.align_phoneme_ids_with_durations(phoneme_ids, durations)

        # aligner
        # use alignment_hard to oversample phonemes
        # Duration Predictor should predict the duration of unmasked phonemes where target is masked alignment_hard

        assert all([exists(el) for el in (phoneme_len, mel_len, phoneme_mask, mel_mask)]), 'need to pass phoneme_len, mel_len, phoneme_mask, mel_mask, to train duration predictor module'

        alignment_hard, _, alignment_logprob, _ = self.forward_aligner(phoneme_emb, phoneme_mask, mel, mel_mask)
        target = alignment_hard

        if exists(self_attn_mask):
            loss_mask = cond_mask & self_attn_mask
        else:
            loss_mask = self_attn_mask

        if not exists(mask):
            return F.l1_loss(x, target)

        loss = F.l1_loss(x, target, reduction = 'none')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
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
        ff_dropout = 0.,
        time_hidden_dim = None,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout = 0,
        attn_flash = False,
        attn_qk_norm = True,
        num_register_tokens = 16,
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

        self.dim_cond_emb = dim_cond_emb
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
            ff_dropout = ff_dropout,
            attn_dropout= attn_dropout,
            attn_flash = attn_flash,
            attn_qk_norm = attn_qk_norm,
            num_register_tokens = num_register_tokens,
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
        times,
        cond_token_ids,
        self_attn_mask = None,
        cond_drop_prob = 0.1,
        target = None,
        cond = None,
        cond_mask = None
    ):
        # project in, in case codebook dim is not equal to model dimensions

        x = self.proj_in(x)

        if exists(cond):
            cond = self.proj_in(cond)

        cond = default(cond, x)

        # shapes

        batch, seq_len, cond_dim = cond.shape
        assert cond_dim == x.shape[-1]

        # auto manage shape of times, for odeint times

        if times.ndim == 0:
            times = repeat(times, '-> b', b = cond.shape[0])

        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = cond.shape[0])

        # construct conditioning mask if not given

        if self.training:
            if not exists(cond_mask):
                if coin_flip():
                    frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask)
                    cond_mask = mask_from_frac_lengths(seq_len, frac_lengths)
                else:
                    cond_mask = prob_mask_like((batch, seq_len), self.p_drop_prob, self.device)
        else:
            if not exists(cond_mask):
                cond_mask = torch.ones((batch, seq_len), device = cond.device, dtype = torch.bool)

        cond_mask_with_pad_dim = rearrange(cond_mask, '... -> ... 1')

        # as described in section 3.2

        x = x * cond_mask_with_pad_dim
        cond = cond * ~cond_mask_with_pad_dim

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
                cond_emb = rearrange(cond_emb, 'b n d -> b d n')
                cond_emb = interpolate_1d(cond_emb, seq_len)
                cond_emb = rearrange(cond_emb, 'b d n -> b n d')

                if exists(self_attn_mask):
                    self_attn_mask = interpolate_1d(self_attn_mask, seq_len)

        # concat source signal, semantic / phoneme conditioning embed, and conditioning
        # and project

        to_concat = [*filter(exists, (x, cond_emb, cond))]
        embed = torch.cat(to_concat, dim = -1)

        x = self.to_embed(embed)

        x = self.conv_embed(x) + x

        time_emb = self.sinu_pos_emb(times)

        # attend

        x = self.transformer(
            x,
            mask = self_attn_mask,
            adaptive_rmsnorm_cond = time_emb
        )

        x = self.to_pred(x)

        # if no target passed in, just return logits

        if not exists(target):
            return x

        loss_mask = reduce_masks_with_and(cond_mask, self_attn_mask)

        if not exists(loss_mask):
            return F.mse_loss(x, target)

        loss = F.mse_loss(x, target, reduction = 'none')

        loss = reduce(loss, 'b n d -> b n', 'mean')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den

        return loss.mean()

# wrapper for the CNF

def is_probably_audio_from_shape(t):
    return exists(t) and (t.ndim == 2 or (t.ndim == 3 and t.shape[1] == 1))

class ConditionalFlowMatcherWrapper(Module):
    @beartype
    def __init__(
        self,
        voicebox: VoiceBox,
        text_to_semantic: Optional[TextToSemantic] = None,
        duration_predictor: Optional[DurationPredictor] = None,
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

        assert not (not self.condition_on_text and exists(text_to_semantic)), 'TextToSemantic should not be passed in if not conditioning on text'
        assert not (exists(text_to_semantic) and not exists(text_to_semantic.wav2vec)), 'the wav2vec module must exist on the TextToSemantic, if being used to condition on text'

        self.text_to_semantic = text_to_semantic
        self.duration_predictor = duration_predictor

        if self.condition_on_text:
            assert exists(text_to_semantic) ^ exists(duration_predictor), 'you should use either TextToSemantic from Spear-TTS, or DurationPredictor for the text / phoneme to audio alignment, but not both'

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

    def load(self, path, strict = True):
        # return pkg so the trainer can access it
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(pkg['model'], strict = strict)
        return pkg

    @torch.inference_mode()
    def sample(
        self,
        *,
        cond = None,
        texts: Optional[List[str]] = None,
        text_token_ids: Optional[Tensor] = None,
        semantic_token_ids: Optional[Tensor] = None,
        phoneme_ids: Optional[Tensor] = None,
        cond_mask = None,
        steps = 3,
        cond_scale = 1.,
        decode_to_audio = True,
        max_semantic_token_ids = 2048,
        spec_decode = False,
        spec_decode_gamma = 5 # could be higher, since speech is probably easier than text, needs to be tested
    ):
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

        self_attn_mask = None
        cond_token_ids = None

        if self.condition_on_text:
            if exists(self.text_to_semantic) or exists(semantic_token_ids):
                assert not exists(phoneme_ids)

                if not exists(semantic_token_ids):
                    self.text_to_semantic.eval()

                    semantic_token_ids, self_attn_mask = self.text_to_semantic.generate(
                        source = default(text_token_ids, texts),
                        source_type = 'text',
                        target_type = 'speech',
                        max_length = max_semantic_token_ids,
                        return_target_mask = True,
                        spec_decode = spec_decode,
                        spec_decode_gamma = spec_decode_gamma
                    )

                cond_token_ids = semantic_token_ids

            elif exists(self.duration_predictor):
                self.duration_predictor.eval()

                durations, aligned_phoneme_ids = self.duration_predictor.forward_with_cond_scale(
                    cond = cond,
                    texts = texts,
                    phoneme_ids = phoneme_ids,
                    return_aligned_phoneme_ids = True
                )

                cond_token_ids = aligned_phoneme_ids

            cond_tokens_seq_len = cond_token_ids.shape[-1]

            if exists(cond):
                if exists(self.text_to_semantic):
                    # calculate the correct conditioning length for text to semantic
                    # based on the sampling freqs of wav2vec and audio-enc-dec, as well as downsample factor
                    # (cond_time x cond_sampling_freq / cond_downsample_factor) == (audio_time x audio_sampling_freq / audio_downsample_factor)
                    wav2vec = self.text_to_semantic.wav2vec
                    audio_enc_dec = self.voicebox.audio_enc_dec

                    cond_target_length = (cond_tokens_seq_len * wav2vec.target_sample_hz / wav2vec.downsample_factor) / (audio_enc_dec.sampling_rate / audio_enc_dec.downsample_factor)
                    cond_target_length = math.ceil(cond_target_length)

                elif exists(self.duration_predictor):
                    cond_target_length = cond_tokens_seq_len

                cond = curtail_or_pad(cond, cond_target_length)
            else:
                cond = torch.zeros((cond_token_ids.shape[0], cond_target_length, self.dim_cond_emb), device = self.device)
        else:
            assert num_cond_inputs == 0, 'no conditioning inputs should be given if not conditioning on text'

        shape = cond.shape
        batch = shape[0]

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
                cond_mask = cond_mask,
                self_attn_mask = self_attn_mask
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
        mask = None,
        semantic_token_ids = None,
        phoneme_ids = None,
        cond = None,
        cond_mask = None,
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

                if exists(cond) and cond_is_raw_audio:
                    cond = resample(cond, input_sampling_rate, audio_enc_dec_sampling_rate)
                    cond = self.voicebox.audio_enc_dec.encode(cond)

        # setup text conditioning, either coming from duration model (as phoneme ids)
        # or from text-to-semantic module, semantic ids encoded with wav2vec (hubert usually)

        assert self.condition_on_text or not (exists(semantic_token_ids) or exists(phoneme_ids)), 'semantic or phoneme ids should not be passed in if not conditioning on text'

        cond_token_ids = None

        if self.condition_on_text:
            if exists(self.text_to_semantic) or exists(semantic_token_ids):
                assert not exists(phoneme_ids), 'phoneme ids are not needed for conditioning with spear-tts text-to-semantic'

                if not exists(semantic_token_ids):
                    assert input_is_raw_audio
                    wav2vec = self.text_to_semantic.wav2vec
                    wav2vec_input = resample(raw_audio, input_sampling_rate, wav2vec.target_sample_hz)
                    semantic_token_ids = wav2vec(wav2vec_input).clone()

                cond_token_ids = semantic_token_ids
            else:
                assert exists(phoneme_ids)
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
            cond_mask = cond_mask,
            times = times,
            target = flow,
            self_attn_mask = mask,
            cond_token_ids = cond_token_ids,
            cond_drop_prob = self.cond_drop_prob
        )

        return loss
