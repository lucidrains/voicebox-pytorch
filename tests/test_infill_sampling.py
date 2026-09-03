import pytest
import torch
from torch import nn


try:
    import voicebox_pytorch.voicebox_pytorch as voicebox_module
except (ImportError, OSError) as exc:
    pytest.skip(f'optional audio dependencies are unavailable: {exc}', allow_module_level=True)


class DummyVoiceBox(nn.Module):
    condition_on_text = False
    audio_enc_dec = None

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.inputs = []

    def forward_with_cond_scale(self, x, **kwargs):
        self.inputs.append(x.detach().clone())
        return torch.ones_like(x)


def euler(fn, y0, times, **kwargs):
    states = [y0]
    y = y0
    for t_start, t_end in zip(times[:-1], times[1:]):
        y = y + (t_end - t_start) * fn(t_start, y)
        states.append(y)
    return torch.stack(states)


def make_wrapper():
    wrapper = object.__new__(voicebox_module.ConditionalFlowMatcherWrapper)
    nn.Module.__init__(wrapper)
    wrapper.voicebox = DummyVoiceBox()
    wrapper.condition_on_text = False
    wrapper.text_to_semantic = None
    wrapper.duration_predictor = None
    wrapper.use_torchode = False
    wrapper.odeint_kwargs = {}
    return wrapper


def test_infill_sampling_keeps_observed_context_fixed(monkeypatch):
    wrapper = make_wrapper()
    cond = torch.arange(12, dtype = torch.float32).reshape(1, 4, 3)
    cond_mask = torch.tensor([[True, False, True, False]])

    monkeypatch.setattr(voicebox_module, 'odeint', euler)
    monkeypatch.setattr(voicebox_module.torch, 'randn_like', torch.zeros_like)

    sampled = wrapper.sample(
        cond = cond,
        cond_mask = cond_mask,
        steps = 3,
        decode_to_audio = False
    )

    observed = ~cond_mask
    torch.testing.assert_close(sampled[observed], cond[observed])
    assert torch.all(sampled[cond_mask] > 0.)

    # The denoiser also receives the projected state, so numerical solver
    # evaluations cannot leak changes into observed frames.
    assert len(wrapper.voicebox.inputs) == 2
    for model_input in wrapper.voicebox.inputs:
        torch.testing.assert_close(model_input[observed], cond[observed])


def test_infill_mask_must_align_with_conditioning_sequence():
    wrapper = make_wrapper()
    cond = torch.zeros(1, 4, 2)

    with pytest.raises(AssertionError, match = 'cond_mask'):
        wrapper.sample(
            cond = cond,
            cond_mask = torch.ones(1, 3, dtype = torch.bool),
            steps = 2,
            decode_to_audio = False
        )
