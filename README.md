# Privacy-Preserving Low-Rank Adaptation for Latent Diffusion Models

This repository contains the code implementations of PrivateLoRA and Stable PrivateLoRA. The implementation is built upon [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts.git).

## Overview
Low-rank adaptation (LoRA) is an efficient strategy for adapting latent diffusion models (LDMs) on a training dataset to generate specific objects by minimizing the adaptation loss. However, adapted LDMs via LoRA are vulnerable to membership inference (MI) attacks that can judge whether a particular data point belongs to private training datasets, thus facing severe risks of privacy leakage. To defend against MI attacks, we make the first effort to propose privacy-preserving LoRA (PrivateLoRA). PrivateLoRA is formulated as a min-max optimization problem where a proxy attack model is trained by maximizing its MI gain while the LDM is adapted by minimizing the sum of the adaptation loss and the proxy attack model's MI gain. 
However, we empirically disclose that PrivateLoRA has the issue of unstable optimization due to the large fluctuation of the gradient scale which impedes adaptation. To mitigate this issue, we propose Stable PrivateLoRA that adapts the LDM by minimizing the ratio of the adaptation loss to the MI gain, which implicitly rescales the gradient and thus stabilizes the optimization. Our comprehensive empirical results corroborate that adapted LDMs via Stable PrivateLoRA can effectively defend against MI attacks while generating high-quality images. 

## Let's start

### PrivateLoRA: PLoRA.py  PLoRA.sh

### Stable PrivateLora: SPLoRA.py  SPLoRA.sh

### Dataset: [Pokemon Dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions), [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

```bibtex
@article{luo2024privacy,
  title={Privacy-Preserving Low-Rank Adaptation for Latent Diffusion Models},
  author={Luo, Zihao and Xu, Xilie and Liu, Feng and Koh, Yun Sing and Wang, Di and Zhang, Jingfeng},
  journal={arXiv preprint arXiv:2402.11989},
  year={2024}
}
