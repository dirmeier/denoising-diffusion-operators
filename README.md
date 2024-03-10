# Denoising diffusion operators

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/ddo/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/ddo/actions/workflows/ci.yaml)

> Implementation of 'Score-based Diffusion Models in Function Space'

## About

This repository implements the method proposed in [Score-based Diffusion Models in Function Space](https://arxiv.org/abs/2302.07400), i.e., 
a function-space version of diffusion probabilistic models, using JAX and Flax.

> [!IMPORTANT]  
> The implementation does not strictly follow the original paper. Specifically, the U-net neural operator ([U-NO](https://arxiv.org/abs/2204.11127)) as well as the sampling are customized and simplified.
> Our U-NO implementation just uses spectral convolutions for up- and down-sampling of input dimensions. 
> We use the VP-parameterization of [DDPM](https://arxiv.org/abs/2006.11239); hence we don't use the score-matching loss in [NCSN](https://arxiv.org/abs/1907.05600) but a conventional SSE. 
> We consequently don't use Langevin dynamics for sampling, but the sampling proposed in DDPM.
> 
> If you find bugs, please open an issue and report them.

## Example usage

The `experiments` folder contains a use case on MNIST-SDF. For training on 32x32-dimensional images from the MNIST-SDF dataset, call:

```bash
cd experiments/mnist_sdf
python main.py --mode=train --epochs=1000
```

Then, in order to sample, call:

```bash
cd experiments/mnist_sdf
python main.py --mode=sample
```

This samples 32x32-, 64x64- and 128x128-dimensional images and creates some figures in `experiments/mnist_sdf/figures`.


## Installation

To install the latest GitHub <TAG>, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/ddo@<TAG>
```

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
