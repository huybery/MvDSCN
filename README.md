# MvDSCN
:game_die: Tensorflow Repo for "Multi-view Deep Subspace Clustering Networks"


[[Paper]](https://arxiv.org/abs/1908.01978) (submitted to **TIP 2019**)

# Overview

In this work, we propose a novel multi-view deep subspace clustering network (MvDSCN) by learning a multi-view self-representation matrix in an end to end manner. 
MvDSCN consists of two sub-networks, i.e., diversity network (Dnet) and universality network (Unet). 
A latent space is built upon deep convolutional auto-encoders and a self-representation matrix is learned in the latent space using a fully connected layer. 
Dnet learns view-specific self-representation matrices while Unet learns a common self-representation matrix for all views. 
To exploit the complementarity of multi-view representations, Hilbert Schmidt Independence Criterion (HSIC) is introduced as a diversity regularization, which can capture
the non-linear and high-order inter-view relations. 
As different views share the same label space, the self-representation matrices of each view are aligned to the common one by a universality regularization.


![MvDSCN](/assets/Architecture.jpg)


# Requirements

* Tensorflow 
* scipy
* numpy
* sklearn
* munkres

# Usage

*  Test by Released Result:

```python
python main.py --test
```

*  Train Network with Finetune.

We have released the pretrain model in `/pretrain` folder, you can train it with finetune: 

```python
python main.py --ft
```

* Pretrain Auoencoder From Scratch:

You re-train autoencoder from scarath:
```
python main.py
```

# Citation
Please star :star2: this repo and cite :page_facing_up: this paper if you want to use it in your work.

```
@article{zhu2019multiview,
    title={Multi-view Deep Subspace Clustering Networks},
    author={Pengfei Zhu and Binyuan Hui and Changqing Zhang and Dawei Du and Longyin Wen and Qinghua Hu},
    journal={ArXiv: 1908.01978}
    year={2019}
}
```

# License
MIT License