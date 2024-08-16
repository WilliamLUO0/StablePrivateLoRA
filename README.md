# Privacy-Preserving Low-Rank Adaptation against Membership Inference Attacks for Latent Diffusion Models

This repository contains the code implementations of PrivateLoRA and Stable PrivateLoRA. The implementation is built upon [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts.git).

## Overview
Low-rank adaptation (LoRA) is an efficient strategy for adapting latent diffusion models (LDMs) on a private dataset to generate specific images by minimizing the adaptation loss. However, the LoRA-adapted LDMs are vulnerable to membership inference (MI) attacks that can judge whether a particular data point belongs to the private dataset, thus leading to the privacy leakage. To defend against MI attacks, we first propose a straightforward solution: Membership-Privacy-preserving LoRA (MP-LoRA). MP-LoRA is formulated as a min-max optimization problem where a proxy attack model is trained by maximizing its MI gain while the LDM is adapted by minimizing the sum of the adaptation loss and the MI gain of the proxy attack model. However, we empirically find that MP-LoRA has the issue of unstable optimization, and theoretically analyze that the potential reason is the unconstrained local smoothness, which impedes the privacy-preserving adaptation. To mitigate this issue, we further propose a Stable Membership-Privacy-preserving LoRA (SMP-LoRA) that adapts the LDM by minimizing the ratio of the adaptation loss to the MI gain. Besides, we theoretically prove that the local smoothness of SMP-LoRA can be constrained by the gradient norm, leading to improved convergence. Our experimental results corroborate that SMP-LoRA can indeed defend against MI attacks and generate high-quality images. 
## Let's start

### MP-LoRA: MP-LoRA-Pokemon.py, MP-LoRA-Pokemon.sh, MP-LoRA-CelebA.py, MP-LoRA-CelebA.sh

### SMP-LoRA: SMP-LoRA-Pokemon.py, SMP-LoRA-Pokemon.sh, SMP-LoRA-CelebA.py, SMP-LoRA-CelebA.sh

### Dataset: [Pokemon Dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions), [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

### Note: Move py file to ./sd-scripts and Run sh file

[//]: # (## Bibtex)

[//]: # (```bibtex)

[//]: # (@article{luo2024privacy,)

[//]: # (  title={Privacy-Preserving Low-Rank Adaptation for Latent Diffusion Models},)

[//]: # (  author={Luo, Zihao and Xu, Xilie and Liu, Feng and Koh, Yun Sing and Wang, Di and Zhang, Jingfeng},)

[//]: # (  journal={arXiv preprint arXiv:2402.11989},)

[//]: # (  year={2024})

[//]: # (})
