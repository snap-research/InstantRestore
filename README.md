# InstantRestore: Single-Step Personalized Face Restoration with Shared-Image Attention

> Howard Zhang*, Yuval Alaluf*, Sizhuo Ma, Achuta Kadambi, Jian Wang†, Kfir Aberman†  
> *Denotes equal contribution  
> †Denotes equal advising
> 
> Face image restoration aims to enhance degraded facial images while addressing challenges such as diverse degradation types, real-time processing demands, and, most crucially, the preservation of identity-specific features. Existing methods often struggle with slow processing times and suboptimal restoration, especially under severe degradation, failing to accurately reconstruct finer-level identity details. To address these issues, we introduce InstantRestore, a novel framework that leverages a single-step image diffusion model and an attention-sharing mechanism for fast and personalized face restoration. Additionally, InstantRestore incorporates a novel landmark attention loss, aligning key facial landmarks to refine the attention maps, enhancing identity preservation. At inference time, given a degraded input and a small (${\sim}4$) set of reference images, InstantRestore performs a single forward pass through the network to achieve near real-time performance. Unlike prior approaches that rely on full diffusion processes or per-identity model tuning, InstantRestore offers a scalable solution suitable for large-scale applications.  Extensive experiments demonstrate that InstantRestore outperforms existing methods in quality and speed, making it an appealing choice for identity-preserving face restoration.

<a href="https://arxiv.org/abs/2412.06753"><img src="https://img.shields.io/badge/arXiv-2412.06753-b31b1b.svg"></a>
<a href="https://snap-research.github.io/InstantRestore/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 

<p align="center">
<img src="docs/teaser.jpg" width="800px"/>  
<br>
Given severely degraded face images, previous diffusion-based models struggle to accurately preserve the input identity. Although existing personalized methods better preserve the input identity, they are computationally expensive and often require per-identity model fine-tuning at test time, making them difficult to scale. In contrast, our model, InstantRestore, efficiently attains improved identity preservation with near-real-time performance.
</p>

# Description
Official implementation of our InstantRestore face restoration paper.

<p align="center">
<img src="docs/comparisons_1.jpeg" width="800px"/>  
<br>
<p align="center">
<img src="docs/comparisons_2.jpeg" width="800px"/>  
<br>
<p align="center">
<img src="docs/results.jpg" width="800px"/>  
<br>

## Setup 
A conda environment containing the necessary libraries is provided in `environment.yaml`.

### Pretrained Checkpoints
You can download the pretrained models here:
- [Base Model](https://huggingface.co/hwdz15508/InstantRestore/blob/main/base_ablation_ckpt.pt)
- [AdaIn Model](https://huggingface.co/hwdz15508/InstantRestore/blob/main/adain_ablation_ckpt.pt)
- [Landmark Attention Model](https://huggingface.co/hwdz15508/InstantRestore/blob/main/lmattn_ablation_ckpt.pt)
- [Final Model](https://huggingface.co/hwdz15508/InstantRestore/blob/main/final_model_ckpt.pt)

### External Models
External models can be found here:
- [External Model Link](https://huggingface.co/hwdz15508/InstantRestore/tree/main)

## Training 
Training currently can be done by running `train.py` in `scripts` folder and passing a config file. For example:
```bash
python train.py --config_path config_files/train.yaml
```
All arguments are parsed using [pyrallis](https://github.com/eladrich/pyrallis) via a dataclass (see `face_replace/configs/train_config.py`) for all the possible arguments.

`train_base.yaml` is used to train a base model without AdaIn or Landmark Attention Loss
`train_landmarkloss_adain.yaml` is used to train a model with AdaIn and Landmark Attention Loss

For training, you need to provide a dataset that has this structure:
    ```
    dataset_folder/
    ├── identity_1/
    │   ├── cropped_images/
    │   │    ├── image1.jpg/png
    │   │    ├── image1.jpg/png
    │   │    └── ...
    ├── identity_2/
    │   ├── cropped_images/
    │   │    ├── image1.jpg/png
    │   │    ├── image1.jpg/png
    │   │    └── ...
    └── ...
    ```

Currently, the code to provide landmarks for the Landmark Attention Loss is not available, so only base and adain training is available.

The code already supports multi-gpu training via `accelerate`. To run multi-gpu training, you can run: 
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch train.py --config_path config_files/train.yaml
```

## Inference
For running inference on a trained model, you can run the `face_replace\inference\test.py`. Make sure to provide a ckpt (download one of the provided checkpoints).

base_ablation_ckpt.pt: checkpoint with the base model (no adain or landmark attention supervision)
adain_ablation_ckpt.pt: checkpoint with adain turned on
lmattn_ablation_ckpt.pt: checkpoint with the landmark attention supervision
final_model_ckpt.pt: checkpoint with both adain and landmark attention supervision on

Also provide a path to the dataset in the `test.py`. It should be in this format:

dataset_folder/
    ├── identity_1/
    │   ├── degraded.png
    │   ├── gt.png
    │   ├── conditioning/
    │   │    ├── 1.png
    │   │    ├── 2.png
    │   │    └── ...
    ├── identity_2/
    │   ├── degraded.png
    │   ├── gt.png
    │   ├── conditioning/
    │   │    ├── 1.png
    │   │    ├── 2.png
    │   │    └── ...
    └── ...

Also provide a path to where to store the results.

## Gradio
Also, you can run the gradio demo with ``gradio_demo.py``

## Contact Info 
Something not working? The code has annoying bugs? Feel free to reach out to Yuval Alaluf or Howard Zhang for help!

# Citation
If you use this code for your research, please cite the following work:
```
@inproceedings{zhang2025instantrestore,
  title={Instantrestore: Single-step personalized face restoration with shared-image attention},
  author={Zhang, Howard and Alaluf, Yuval and Ma, Sizhuo and Kadambi, Achuta and Wang, Jian and Aberman, Kfir},
  booktitle={Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  pages={1--10},
  year={2025}
}
```
