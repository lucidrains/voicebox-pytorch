<img src="./images/voicebox.png" width="400px"></img>

## Voicebox - Pytorch

Implementation of <a href="https://arxiv.org/abs/2306.15687">Voicebox</a>, new SOTA Text-to-Speech model from MetaAI, in Pytorch. <a href="https://about.fb.com/news/2023/06/introducing-voicebox-ai-for-speech-generation/">Press release</a>

In this work, we will use rotary embeddings. The authors seem unaware that ALiBi cannot be straightforwardly used for bidirectional models.

The paper also addresses the issue with time embedding incorrectly subjected to relative distances (they concat the time embedding along the frame dimension of the audio tokens). This repository will use adaptive normalization, as applied successfully in <a href="https://arxiv.org/abs/2211.07292">Paella</a>

## Appreciation

- <a href="https://translated.com"><img style="vertical-align: middle;" src="./images/translated.png" height="20px" alt="Translated"><img></a> for awarding me the <a href="https://imminent.translated.com/research-grants-ceremony-innovations-in-language-technology">Imminent Grant</a> to advance the state of open sourced text-to-speech solutions. This project was started and will be completed under this grant.

- <a href="https://stability.ai/">StabilityAI</a> for the generous sponsorship, as well as my other sponsors, for affording me the independence to open source artificial intelligence.

- <a href="https://github.com/b-chiang">Bryan Chiang</a> for the ongoing code review, sharing his expertise on TTS, and pointing me to <a href="https://github.com/atong01/conditional-flow-matching">an open sourced implementation</a> of conditional flow matching

- <a href="https://github.com/manmay-nakhashi">Manmay</a> for getting the repository started with the alignment code

- <a href="https://github.com/chenht2010">@chenht2010</a> for finding a bug with rotary positions, and for validating that the code in the repository converges

- <a href="https://github.com/lucasnewman">Lucas Newman</a> for (yet again) pull requesting all the training code for Spear-TTS conditioned Voicebox training!

- <a href="https://github.com/lucasnewman">Lucas Newman</a> has demonstrated that the whole system works with Spear-TTS conditioning. Training converges even better than <a href="https://github.com/lucidrains/soundstorm-pytorch">Soundstorm</a>

## Install

```bash
$ pip install voicebox-pytorch
```

## Usage

Training and sampling with `TextToSemantic` module from <a href="https://github.com/lucidrains/spear-tts-pytorch">SpearTTS</a>

```python
import torch

from voicebox_pytorch import (
    VoiceBox,
    EncodecVoco,
    ConditionalFlowMatcherWrapper,
    HubertWithKmeans,
    TextToSemantic
)

# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert

wav2vec = HubertWithKmeans(
    checkpoint_path = '/path/to/hubert/checkpoint.pt',
    kmeans_path = '/path/to/hubert/kmeans.bin'
)

text_to_semantic = TextToSemantic(
    wav2vec = wav2vec,
    dim = 512,
    source_depth = 1,
    target_depth = 1,
    use_openai_tokenizer = True
)

text_to_semantic.load('/path/to/trained/spear-tts/model.pt')

model = VoiceBox(
    dim = 512,
    audio_enc_dec = EncodecVoco(),
    num_cond_tokens = 500,
    depth = 2,
    dim_head = 64,
    heads = 16
)

cfm_wrapper = ConditionalFlowMatcherWrapper(
    voicebox = model,
    text_to_semantic = text_to_semantic
)

# mock data

audio = torch.randn(2, 12000)

# train

loss = cfm_wrapper(audio)
loss.backward()

# after much training

texts = [
    'the rain in spain falls mainly in the plains',
    'she sells sea shells by the seashore'
]

cond = torch.randn(2, 12000)
sampled = cfm_wrapper.sample(cond = cond, texts = texts) # (2, 1, <audio length>)
```

For unconditional training, `condition_on_text` on `VoiceBox` must be set to `False`

```python
import torch
from voicebox_pytorch import (
    VoiceBox,
    ConditionalFlowMatcherWrapper
)

model = VoiceBox(
    dim = 512,
    num_cond_tokens = 500,
    depth = 2,
    dim_head = 64,
    heads = 16,
    condition_on_text = False
)

cfm_wrapper = ConditionalFlowMatcherWrapper(
    voicebox = model
)

# mock data

x = torch.randn(2, 1024, 512)

# train

loss = cfm_wrapper(x)

loss.backward()

# after much training

cond = torch.randn(2, 1024, 512)

sampled = cfm_wrapper.sample(cond = cond) # (2, 1024, 512)
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
- [x] setup training and sampling with raw audio, if `audio_enc_dec` is passed in
- [x] integrate with log mel spec / encodec - vocos
- [x] spear-tts-integration
- [x] basic accelerate trainer - thanks to @lucasnewman!

- [ ] cleanup NS2 aligner class and then setup duration predictor training
- [ ] figure out the correct settings for `MelVoco` encode, as the reconstructed audio is longer in length
- [ ] calculate how many seconds corresponds to each frame and add as property on `AudioEncoderDecoder` - when sampling, allow for specifying in seconds

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

```bibtex
@misc{darcet2023vision,
    title   = {Vision Transformers Need Registers},
    author  = {Timothée Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
    year    = {2023},
    eprint  = {2309.16588},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@inproceedings{Dehghani2023ScalingVT,
    title   = {Scaling Vision Transformers to 22 Billion Parameters},
    author  = {Mostafa Dehghani and Josip Djolonga and Basil Mustafa and Piotr Padlewski and Jonathan Heek and Justin Gilmer and Andreas Steiner and Mathilde Caron and Robert Geirhos and Ibrahim M. Alabdulmohsin and Rodolphe Jenatton and Lucas Beyer and Michael Tschannen and Anurag Arnab and Xiao Wang and Carlos Riquelme and Matthias Minderer and Joan Puigcerver and Utku Evci and Manoj Kumar and Sjoerd van Steenkiste and Gamaleldin F. Elsayed and Aravindh Mahendran and Fisher Yu and Avital Oliver and Fantine Huot and Jasmijn Bastings and Mark Collier and Alexey A. Gritsenko and Vighnesh Birodkar and Cristina Nader Vasconcelos and Yi Tay and Thomas Mensink and Alexander Kolesnikov and Filip Paveti'c and Dustin Tran and Thomas Kipf and Mario Luvci'c and Xiaohua Zhai and Daniel Keysers and Jeremiah Harmsen and Neil Houlsby},
    booktitle = {International Conference on Machine Learning},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:256808367}
}
```
