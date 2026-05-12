
### Struture of `experiments` directory

```
experiments/
├── shared_utils/             # DRY: Common metrics (mIoU), loggers, and data loaders
├── configs/                  # Global experiment YAMLs (Hydra/Argparse configs)
├── data/                     # Symlinks to datasets to keep weights/code separate
│
├── Puzzle_Understanding/
│   ├── run.py                # Script to execute puzzle logic & benchmark solvers
│   └── results/
│
├── Representation_Analysis/
│   ├── run.py                # Wrapper for activation maps & global/patch metrics
│   ├── qualitative/
│   │   ├── baselines/
│   │   │   └── activation-maps/
│   │   │       ├── clean-dift/
│   │   │       ├── clip-vit-l-14/
│   │   │       ├── dino-v2/
│   │   │       ├── dino-v3/
│   │   │       ├── sd-1.5/   # Isolated: distinct latent space from 2.1
│   │   │       └── sd-2.1/   # Isolated: distinct architecture
│   │   └── ddn/
│   │       └── activation-maps/
│   │           ├── ddn-sd-1.5/
│   │           └── ddn-sd-2.1/
│   └── quantitative/
│       ├── global/
│       │   └── plots/        # Flattened: results stored directly as visuals
│       └── patch/
│           └── plots/
│
├── Segmentation/
│   ├── run_eval.py           # Entry point for evaluating specific checkpoints
│   ├── run_train.py          # Entry point for starting/resuming training
│   ├── evaluation/
│   │   ├── src/              # Evaluation logic (mIoU, Pixel Acc)
│   │   └── results/
│   └── training/
│       ├── src/              # Model definitions and loss functions
│       ├── checkpoints/      # .pth / .ckpt storage
│       └── logs/             # Tensorboard/WandB files
│           ├── round-1/
│           └── round-2/
│
└── Vision_Language_Alignment/
    ├── run_eval.py           # Benchmarks: classification, retrieval, and seg
    ├── run_train.py          # Alignment/contrastive learning entry point
    ├── evaluation/
    │   ├── src/
    │   ├── prompts/          # Centralized templates (e.g., CuPL, zero-shot)
    │   └── results/          
    │       ├── classification/
    │       │   ├── baselines/
    │       │   │   ├── clip-vit-l-14/
    │       │   │   └── tdino-v2/
    │       │   ├── tdn/
    │       │   │   ├── zero-shot-cupl/
    │       │   │   └── few-shot-tip/
    │       │   └── tddn/
    │       │       ├── zero-shot-cupl/
    │       │       └── few-shot-tip/
    │       ├── retrieval/
    │       └── segmentation/
    └── training/
        ├── src/              # Contrastive learning / Alignment logic
        ├── checkpoints/
        └── stats/            # Markdown reports and training curves
```