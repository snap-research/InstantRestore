## Setup 
A conda environment containing the necessary libraries is provided in `environment.yaml`.

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