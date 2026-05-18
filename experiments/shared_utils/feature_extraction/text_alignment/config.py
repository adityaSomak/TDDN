import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlignConfig:
    # --- Model ---
    embed_dim: int = 512

    # Vision backbone: either HF model ID or config+weights path
    vision_backbone_hf_model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    vision_backbone_config: Optional[str] = None
    vision_backbone_pretrained_weights: Optional[str] = None
    vision_model_train_img_size: int = 224
    vision_num_head_blocks: int = 2
    head_blocks_drop_path: float = 0.3
    use_rope_in_head: bool = True
    use_linear_projection: bool = False

    # Text encoder
    text_encoder_name: str = "ViT-B-16"
    text_encoder_pretrained: str = "openai"
    text_layer_idx: int = 24
    text_num_head_blocks: int = 2
    text_head_blocks_drop_path: float = 0.1

    init_logit_scale: float = math.log(1 / 0.07)
    freeze_logit_scale: bool = False
    label_smoothing: float = 0.0

    # --- CLIP Loss ---
    use_simple_clip_loss: bool = True
    clip_temperature: float = 0.05
    normalize_clip_embeddings: bool = True

    # --- Gram Loss ---
    use_gram_loss: bool = True
    gram_loss_weight: float = 1.0
    patch_sampling_rate: float = 1.0
    normalize_patches: bool = True

    # --- Structure Loss (STRUCTURE defaults) ---
    use_structure_loss: bool = True
    structure_lambda: float = 10.0
    structure_temperature: float = 0.05
    structure_levels: int = 1
    structure_weighting: str = "none"
    structure_warmup_steps: int = 1000
    structure_margin: float = 0.0
    structure_centering: str = "mean"
    structure_distance: str = "cosine"
    structure_center_first: bool = False

    # --- KD Loss (frozen teacher anchoring, anti-catastrophic-forgetting) ---
    use_kd_loss: bool = False
    kd_lambda: float = 1.0
    kd_checkpoint: Optional[str] = None  # pretrained teacher ckpt; defaults to resume_checkpoint

    # --- Optimizer ---
    lr: float = 5e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    gradient_clip: Optional[float] = None

    # --- Schedule ---
    max_iterations: int = 50000
    warmup_iterations: int = 2000

    # --- Data ---
    batch_size: int = 512
    gradient_accumulation_steps: int = 1
    num_workers: int = 10
    coco_root: str = ""
    coco_ann_file: str = ""
    laion_shards: str = ""
    coco_laion_ratio: float = 0.5
    max_laion_samples: Optional[int] = None

    # --- Parallelization ---
    param_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    use_fsdp: bool = True
    use_ac: bool = True
    do_compile: bool = True

    # --- Eval paths (optional) ---
    flickr_root: str = ""
    flickr_ann_file: str = ""
    flickr_split: str = "test"
    coco_val_root: str = ""
    coco_val_ann_file: str = ""
    cifar100_root: str = ""
    use_extended_prompts: bool = True

    # --- Checkpointing / Logging ---
    output_dir: str = "output"
    checkpointing_period: int = 500
    eval_freq: int = 5000
    gc_freq: int = 100
    max_checkpoints_to_keep: int = 5
    log_freq: int = 10
    resume: bool = True
    resume_checkpoint: Optional[str] = None
    pretrained_weights: Optional[str] = None

    # --- Training mode ---
    training_mode: str = "auto"          # "auto", "live", "features"
    grad_cache_multiplier: int = 8       # micro-batches accumulated for GradCache (live mode)

    seed: int = 11

    # --- CleanDIFT backbone (use instead of DINOv3 when use_cleandift=True) ---
    use_cleandift: bool = False
    cleandift_proj_dim: int = 512        # per-layer MLP output dim; embed_dim = 3 × this
    cleandift_common_grid: int = 21      # interpolation target (21×21=441 tokens @ 336px)
    cleandift_pca_dir: str = ""          # path to pca/ dir with cd_layer{l}_mean/components.npy
    cleandift_pca_dim: int = 512         # PCA components per layer
    cleandift_use_cls: bool = False      # learnable CLS token; output = cat([CLS, mean(patches)]) → 2×head_dim
    use_fused_encoder: bool = False      # DINOv3 ViT-H + CleanDIFT patch fusion; embed_dim = 2×backbone_dim
