from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List


class SchedulerType(Enum):
    COSINE = 'cosine'
    STEP = 'step'
    LINEAR = 'linear'
    COSINE_WITH_RESTARTS = 'cosine_with_restarts'
    POLYNOMIAL = 'polynomial'
    CONSTANT = 'constant'
    CONSTANT_WITH_WARMUP = 'constant_with_warmup'


@dataclass
class ComputeConfig:
    """ Config for resources """
    # Batch size for training
    batch_size: int = field(default=3)
    # Batch size for testing and inference
    test_batch_size: Optional[int] = field(default=None)
    # Number of train dataloader workers
    workers: int = field(default=12)
    # Number of test/inference dataloader workers
    test_workers: Optional[int] = field(default=None)
    # Random seed
    seed: int = field(default=42)

    def __post_init__(self):
        if self.test_batch_size is None:
            self.test_batch_size = self.batch_size
        if self.test_workers is None:
            self.test_workers = self.workers


@dataclass
class OptimConfig:
    """ Config for scheduling, optimization, and losses """
    # which optimizer to use
    optim_name: str = field(default="adamW")
    # Optimizer learning rate
    learning_rate: float = field(default=5e-4)
    # Learning rate scheduler
    scheduler_type: Optional[SchedulerType] = field(default=SchedulerType.COSINE)
    # Target learning rate for learning rate decay
    target_lr: float = field(default=5e-6)
    # Whether to use gradient clipping
    use_clip_grad: bool = field(default=True)
    # Gradient clipping max norm value
    clip_grad_max_norm: float = field(default=1.0)
    # Gradient clipping norm type
    clip_grad_norm_type: float = field(default=2)
    # Weight decay value for optimizer
    weight_decay: float = field(default=1e-2)
    # Whether to use mixed precision
    mixed_precision: bool = field(default=True)
    # Number of gradient accumulation steps
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=False)
    # Args for the loss function from pix2pix turbo
    gan_disc_type: str = field(default="vagan_clip")
    gan_loss_type: str = field(default="multilevel_sigmoid_s")
    lambda_gan: float = field(default=0.5)
    lambda_lpips: float = field(default=5.0)
    lambda_l2: float = field(default=5.0)
    lambda_l1: float = field(default=0.0)
    lambda_ssim: float = field(default=0.0)
    lambda_id_loss: float = field(default=1.0)
    lambda_attn_reg: float = field(default=0.0)
    lambda_clipsim: float = field(default=0.0)
    lambda_dreamsim: float = field(default=0.0)
    lambda_wavelets_loss: float = field(default=0.0)
    lambda_latent_loss: float = field(default=0.0)
    lambda_cycle: float = field(default=0.0)
    lambda_landmark: float = field(default=0.0)
    lambda_pos_reg: float = field(default=0.0)
    lambda_neg_reg: float = field(default=0.0)
    lambda_facial_comp: float = field(default=0.0)
    compute_id_loss_between_identities: bool = field(default=False)
    # Parameters related to the learning rate and optimizer
    lr_warmup_steps: int = field(default=100)
    lr_num_cycles: int = field(default=1)
    lr_power: float = field(default=1.0)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_weight_decay: float = field(default=1e-2)
    adam_epsilon: float = field(default=1e-08)
    # Additional parameters
    enable_xformers_memory_efficient_attention: bool = field(default=False)    

@dataclass
class DataConfig:
    """ Config for training and evaluation data """
    # Type of dataset/experiment to run
    dataset_type: str = field(default="debug")
    # Get the path to the data
    data_root: Union[Path, List[Path]] = field(default=Path("/nfs/usr/yuval/face_replace/data/celeba_v4.1"))
    # Get the path to the val data
    val_data_root: Path = field(default=Path("/nfs/usr/yuval/face_replace/data/myvlm_data_cropped_v4"))
    # Whether to train on a small subset of data
    overfit: bool = field(default=False)
    # Whether to check leakage between train and test
    test_leakage: bool = field(default=True)
    # How to preprocess the data
    train_image_prep: str = field(default="resized_crop_512")
    test_image_prep: str = field(default="resized_crop_512")
    resolution: int = field(default=512)
    # Max number of images to use for conditioning from the same identity
    max_conditioning_images: int = field(default=4)
    # Whether to augment the masks
    augment_masks: bool = field(default=False)
    store_landmarks: bool = field(default=False)


@dataclass
class ModelConfig:
    """ Model configuration """
    net_type: str = field(default="pix2pix_turbo")
    # Whether to use a pretrained backbone clear
    use_pretrained: bool = field(default=True)
    # Model arguments for pix2pix turbo
    lora_rank_unet: int = field(default=16)
    lora_rank_vae: int = field(default=16)
    # Whether to pass the face embeddings instead of the text embeddings to the cross-attention
    condition_on_face_embeds: bool = field(default=False)
    # Whether the mask and landmarks are passed as additional channels to the vae
    concat_mask_and_landmarks: bool = field(default=False)
    # Whether to use shared attention to inject the IDs
    use_shared_attention: bool = field(default=True)
    # The timestep to noise the input images to
    noise_timestep: int = field(default=249)
    # Whether to train the VAE or not 
    train_vae: bool = field(default=True)
    # Whether to train only the VAE encoder or not 
    train_only_vae_encoder: bool = field(default=False)
    # Checkpoint path for resuming training
    checkpoint_path: Optional[Path] = field(default=None)
    # Whether to use shortcuts
    use_shortcuts: bool = field(default=False)
    guidance_scale: float = field(default=0.0)
    train_reference_networks: bool = field(default=False)
    # Whether to use AdaIn layers on values
    use_adain: bool = field(default=False)
    # Whether to concat input keys and values in Shared Attention
    train_input: bool = field(default = True)


@dataclass
class LogConfig:
    """ Config for logging """
    # Directory to save all experiments to
    exp_root: Path = field(default=Path("/nfs/usr/yuval/face_replace/experiments"))
    # Name of current experiment
    exp_name: str = field(default="celeb_v4_with_visualizations")
    # Whether to overwrite an existing experiment
    allow_overwrite: bool = field(default=True)
    # Whether to log experiment to weights and biases
    log2wandb: bool = field(default=True)
    # Number of batches to visualize for validation
    val_vis_count: int = field(default=50)
    # Whether to visualize attention maps
    vis_attention: bool = field(default=True)

    @property
    def exp_dir(self) -> Path:
        return self.exp_root / self.exp_name


@dataclass
class TrainStepsConfig:
    """ Define intervals for validation, evaluation, etc while training"""
    # Maximum number of training steps
    max_steps: int = field(default=15_000)
    # Interval for logging train images during training
    image_interval: int = field(default=150)
    # Interval for logging metrics to log / wandb
    metric_interval: int = field(default=10)
    # Validation interval
    val_interval: int = field(default=250)
    # Model checkpoint interval
    save_interval: int = field(default=100_000)


@dataclass
class TrainConfig:
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log: LogConfig = field(default_factory=LogConfig)
    steps: TrainStepsConfig = field(default_factory=TrainStepsConfig)
