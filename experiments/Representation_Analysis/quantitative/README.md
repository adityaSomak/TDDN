# Quantitative — CKA and representation quality

Two metric families on a fixed 2000-image COCO val2014 subset
(`configs/coco_sample_ids.csv`):

| Family       | Metrics                                | Module                  |
|---|---|---|
| Similarity   | `linear_cka`, `pwcca`                  | `metrics/similarity.py` |
| Quality      | `uniformity`, `effective_rank`         | `metrics/quality.py`    |

## Layout

```
global/
├── results/
│   ├── global_quality.csv             # 8 reps × (uniformity, effective_rank)
│   └── global_similarity.csv          # 11 pairs × (linear_cka, pwcca)
└── plots/
    ├── global_uniformity.{pdf,png}
    ├── global_effective_rank.{pdf,png}
    └── global_similarity_bars.{pdf,png}

patch/
├── results/
│   ├── patch_quality.csv              # 8 reps
│   └── patch_similarity.csv           # 6 pairs
└── plots/
    ├── patch_uniformity.{pdf,png}
    ├── patch_effective_rank.{pdf,png}
    └── patch_similarity_bars.{pdf,png}
```

## Notation

CSV `representation` / `pair` tokens map to LaTeX labels at plot time
(see `GLOBAL_LABEL`, `PATCH_LABEL`, `GLOBAL_PAIR_LABEL`, `PATCH_PAIR_LABEL`
in `run.py`):

| CSV token             | LaTeX            | Meaning                                                  |
|---|---|---|
| `dino(cls)`           | DN_g             | DINOv3 CLS token                                         |
| `dino(mean)`          | DN_p̄            | DINOv3 mean-pooled patch tokens                          |
| `cd(2+5+8)`           | CD_p̄            | CleanDIFT layers 2+5+8 (PCA-reduced, L2-norm, concat)    |
| `sd(2+5+8)`           | SD_p̄            | Stable Diffusion v2.1 layers 2+5+8                       |
| `clip(image)`         | CLIP_g           | CLIP ViT-L/14 image embedding                            |
| `ddn_g`               | DDN              | 0.5·L2(DN) ⊕ 0.5·L2(CD) handcraft fusion                 |
| `fused`               | TDDN             | trained DINOv3 + CleanDIFT alignment head                |
| `vith`                | TDN              | trained DINOv3 + RoBERTa alignment head                  |
| `dino_p`, `cd_p`, ... | DN_p, CD_p, ...  | patch-level variants (same backbones)                    |
| `fused_p`             | DDN_p            | patch-level handcraft fusion                             |
| `fused_trained_p`     | TDDN_p           | patch-level trained TDDN                                 |
| `vith_p`              | TDN_p            | patch-level trained TDN                                  |

The CSVs are committed with the published numbers. `python run.py plots`
re-renders all six figures from them.

Two further analyses live alongside `global/`/`patch/` but are standalone
drivers, not `run.py` subcommands: `alignment/run_alignment.py` (Wang & Isola
alignment) and `cross_prediction/run_cross_prediction.py`
(DINOv3↔CleanDIFT cross-prediction). Each is invoked directly with
`--smoke`/`--full`; see their module docstrings.
