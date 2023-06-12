<!--
Copyright © <2023> Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Florian MAI <florian.ren.mai@googlemail.com>
SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
SPDX-FileContributor: Juan Pablo Zuluaga <juan-pablo.zuluaga@idiap.ch>

SPDX-License-Identifier: MIT
-->
# HyperMixing
HyperMixing is a token-mixing techniques to be used as linear-time alternative to attention, for example in Transformer-like architecture like [HyperMixer](https://arxiv.org/abs/2203.03691).

This repository serves as the unified PyTorch implementation for both [single-head hypermixing](https://arxiv.org/abs/2203.03691) and [multi-head-hypermixing](arxiv.org/abs/123).

![Alt text](figures/hypermixing.png?raw=true "HyperMixing Overview")

# Requirements
Code was tested with:
* Python 3.10
* PyTorch 2.0

You can create an environment with the required dependencies by running

```bash
conda env create -f environment.yml
```

# Installation
```bash
cd hypermixing
pip install .
```

# Usage
```python3
import torch
from hypermixing import HyperMixing

input_dim = 128
hypernet_size = 512
tied = False
num_heads = 2
max_length = 3000
token_mixer = HyperMixing(input_output_dim=input_dim,
        hypernet_size=hypernet_size,
        tied=tied,
        num_heads=num_heads,
        max_length=max_length)

queries = torch.randn((64, 50, 128)) # [bsize, num_queries, emb_dim]
keys = torch.randn((64, 25, 128)) # [bsize, num_keys, emb_dim]
values = torch.randn((64, 25, 128)) # [bsize, num_keys, emb_dim]
out = token_mixer(queries, keys, values) # [bsize, num_queries, emb_dim]
assert out.size() == queries.size()
```

# Citation
If you use or build on HyperMixer, please cite the following papers:

```latex
@inproceedings{mai2023hypermixer,
    author = {Mai, F. and Pannatier, A. and Fehr, F. and Chen, H. and Marelli, F. and Fleuret, F. and Henderson, J.},
    title = {HyperMixer: An MLP-based Low Cost Alternative to Transformers},
    booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    year = {2023}
}

@article{mai2023multihead-hypermixer,
    author={Mai, F. and Zuluaga-Gomez, J. and Parcollet, T. and Motlicek, P.},
    title={HyperConformer: Multi-head HyperMixer for Efficient Speech Recognition},
    booktitle = {Proc. Interspeech 2023},
    year={2023}
}
```
