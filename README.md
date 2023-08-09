<img src="./voicebox.png" width="400px"></img>

## Voicebox - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2306.15687">Voicebox</a>, new SOTA Text-to-Speech model from MetaAI, in Pytorch. <a href="https://about.fb.com/news/2023/06/introducing-voicebox-ai-for-speech-generation/">Press release</a>

In this work, we will use rotary embeddings. The authors seem unaware that ALiBi cannot be straightforwardly used for bidirectional models.

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the generous sponsorship, as well as my other sponsors, for affording me the independence to open source artificial intelligence.

- <a href="https://translated.com">Translated</a> for the generous grant to advance the state of open source text-to-speech solutions. This project was started and will be finished under this grant.

- <a href="https://github.com/b-chiang">Bryan Chiang</a> for the ongoing code review, sharing his expertise on TTS, and pointing me to <a href="https://github.com/atong01/conditional-flow-matching">an open sourced implementation</a> of conditional flow matching

- <a href="https://github.com/manmay-nakhashi">Manmay</a> for getting the repository started with the alignment code

## Install

```bash
$ pip install voicebox-pytorch
```

## Usage

```python
import torch

from voicebox_pytorch import (
    VoiceBox,
    ConditionalFlowMatcherWrapper
)

model = VoiceBox(
    dim = 512,
    num_phoneme_tokens = 256,
    depth = 2,
    dim_head = 64,
    heads = 16
)

cfm_wrapper = ConditionalFlowMatcherWrapper(
    voicebox = model,
    use_torchode = False   # by default will use torchdiffeq with midpoint as in paper, but can use the promising torchode package too
)

x = torch.randn(2, 1024, 512)
phonemes = torch.randint(0, 256, (2, 1024))
mask = torch.randint(0, 2, (2, 1024)).bool()

loss = cfm_wrapper(
    x,
    phoneme_ids = phonemes,
    cond = x,
    mask = mask
)

loss.backward()

# after much training above...

sampled = cfm_wrapper.sample(
    phoneme_ids = phonemes,
    cond = x,
    mask = mask
) # (2, 1024, 512) <- same as cond

```

## Todo

- [x] read and internalize original flow matching paper
    - [x] basic loss
    - [x] get neural ode working with torchdyn
- [x] get basic mask generation logic with the p_drop of 0.2-0.3 for ICL
- [x] take care of p_drop, different between voicebox and duration model
- [x] support torchdiffeq and torchode
- [x] switch to adaptive rmsnorm for time conditioning
- [x] add encodec / voco for starters

- [ ] setup training and sampling with raw audio, if `audio_enc_dec` is passed in
- [ ] calculate how many seconds corresponds to each frame and add as property on `AudioEncoderDecoder` - when sampling, allow for specifying in seconds
- [ ] integrate with either hifi-gan and soundstream / encodec
- [ ] basic trainer

## Citations

```bibtex
@article{Le2023VoiceboxTM,
    title   = {Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale},
    author  = {Matt Le and Apoorv Vyas and Bowen Shi and Brian Karrer and Leda Sari and Rashel Moritz and Mary Williamson and Vimal Manohar and Yossi Adi and Jay Mahadeokar and Wei-Ning Hsu},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2306.15687},
    url     = {https://api.semanticscholar.org/CorpusID:259275061}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@misc{torchdiffeq,
    author  = {Chen, Ricky T. Q.},
    title   = {torchdiffeq},
    year    = {2018},
    url     = {https://github.com/rtqichen/torchdiffeq},
}
```

```bibtex
@inproceedings{lienen2022torchode,
    title     = {torchode: A Parallel {ODE} Solver for PyTorch},
    author    = {Marten Lienen and Stephan G{\"u}nnemann},
    booktitle = {The Symbiosis of Deep Learning and Differential Equations II, NeurIPS},
    year      = {2022},
    url       = {https://openreview.net/forum?id=uiKVKTiUYB0}
}
```

```bibtex
@article{siuzdak2023vocos,
    title   = {Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis},
    author  = {Siuzdak, Hubert},
    journal = {arXiv preprint arXiv:2306.00814},
    year    = {2023}
}
```
