<img src="./voicebox.png" width="400px"></img>

## Voicebox - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2306.15687">Voicebox</a>, new SOTA Text-to-Speech model from MetaAI, in Pytorch. <a href="https://about.fb.com/news/2023/06/introducing-voicebox-ai-for-speech-generation/">Press release</a>

In this work, we will use an alternative to ALiBi positional encoding. The authors seem unaware that ALiBi cannot be straightforwardly used for bidirectional models.

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
