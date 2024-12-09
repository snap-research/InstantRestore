# InstantRestore: Single-Step Personalized Face Restoration with Shared-Image Attention

> Howard Zhang*, Yuval Alaluf*, Sizhuo Ma, Achuta Kadambi, Jian Wang†, Kfir Aberman†  
> *Denotes equal contribution  
> †Denotes equal advising
> 
> Face image restoration aims to enhance degraded facial images while addressing challenges such as diverse degradation types, real-time processing demands, and, most crucially, the preservation of identity-specific features. Existing methods often struggle with slow processing times and suboptimal restoration, especially under severe degradation, failing to accurately reconstruct finer-level identity details. To address these issues, we introduce InstantRestore, a novel framework that leverages a single-step image diffusion model and an attention-sharing mechanism for fast and personalized face restoration. Additionally, InstantRestore incorporates a novel landmark attention loss, aligning key facial landmarks to refine the attention maps, enhancing identity preservation. At inference time, given a degraded input and a small (${\sim}4$) set of reference images, InstantRestore performs a single forward pass through the network to achieve near real-time performance. Unlike prior approaches that rely on full diffusion processes or per-identity model tuning, InstantRestore offers a scalable solution suitable for large-scale applications.  Extensive experiments demonstrate that InstantRestore outperforms existing methods in quality and speed, making it an appealing choice for identity-preserving face restoration.

<a href="https://snap-research.github.io/InstantRestore/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 

<p align="center">
<img src="docs/teaser.jpg" width="800px"/>  
<br>
Given severely degraded face images, previous diffusion-based models struggle to accurately preserve the input identity. Although existing personalized methods better preserve the input identity, they are computationally expensive and often require per-identity model fine-tuning at test time, making them difficult to scale. In contrast, our model, InstantRestore, efficiently attains improved identity preservation with near-real-time performance.
</p>

# Description
Official implementation of our InstantRestore face restoration paper.

# Code Coming Soon!
<p align="center">
<img src="docs/comparisons_1.jpeg" width="800px"/>  
<br>
<p align="center">
<img src="docs/comparisons_2.jpeg" width="800px"/>  
<br>
<p align="center">
<img src="docs/results.jpg" width="800px"/>  
<br>

# Citation
If you use this code for your research, please cite the following work:
```

```