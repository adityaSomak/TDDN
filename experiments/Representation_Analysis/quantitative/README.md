# Quantitative — CKA and representation quality

Two metric families on a fixed 2000-image COCO val2014 subset
(`configs/coco_sample_ids.csv`):

| Family       | Metrics                                | Module                  |
|---|---|---|
| Similarity   | `linear_cka`, `pwcca`, `svcca`         | `metrics/similarity.py` |
| Quality      | `uniformity`, `effective_rank`         | `metrics/quality.py`    |

## Layout

```
global/
├── results/
│   ├── global_quality.csv          # representation, uniformity, effective_rank
│   └── global_similarity.csv       # pair, linear_cka, pwcca
└── plots/
    ├── global_uniformity.{pdf,png}
    ├── global_effective_rank.{pdf,png}
    └── global_similarity_bars.{pdf,png}

patch/
├── results/
│   ├── patch_quality.csv
│   └── patch_similarity.csv
└── plots/
    ├── patch_uniformity.{pdf,png}
    └── patch_effective_rank.{pdf,png}
```

## Notation

| Symbol  | Meaning                                                            |
|---|---|
| `DN`, `DN_p`    | DINOv3 ViT-H last-layer patch tokens                       |
| `DN_g`          | DINOv3 mean-pooled global vector                           |
| `CD`, `CD_p`    | CleanDIFT layers 2 + 5 + 8 (PCA-reduced, L2-norm, concat)  |
| `DDN`, `DDN_g`  | 0.5·L2(DN) ⊕ 0.5·L2(CD) — handcraft fusion                 |
| `TDDN_g`        | Trained DINOv3 + CleanDIFT alignment head                  |

The CSVs are committed with the published numbers. To refresh just the
figures after editing a CSV: `python run.py plots`.
