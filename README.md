# Elastic Sparse Mixture of Experts TinyLlama (ElasticMoE)

Sparse Mixture of Experts Transformers have shown promise at the frontier of language modeling. While many state-of-the-art models are currently implemented as sparse mixture of expert models, most models proceed with a fixed expert count. In this work, we present the Elastic Sparse MoE (eMoE) Transformer. In eMoE, experts are shared between layers, and the pool of experts is sized dynamically utilizing curriculum learning.

# Introduction

In a standard Sparse MOE Transformer, the MLP layers inside each layer are duplicated N-fold. During the feed forward pass, the word token coming out of the attention block is passed only through one or a few of these MLPs. Since there are 8 choices in every attention block, the attention blocks are also endowed by a small classifier on top of the word token. This classifier is called the ‘router’, it too is a small MLP.

# Installation

Setup the environment

```
pip install -r requirements.txt
``

Run the notebook in `proto.ipynb`
