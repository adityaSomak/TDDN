# Vision_Language_Alignment

Image-text alignment evaluation across three tasks, run against three
encoder variants:

| Task | Datasets | Modes |
|---|---|---|
| **Classification** | CIFAR-100, Caltech-101, Food-101, GTSRB, ImageNet-1K | zero-shot template, zero-shot CuPL, few-shot TIP-Adapter (K ∈ {1, 2, 4, 8, 16}) |
| **Retrieval** | Flickr30K, COCO val2014 | bidirectional Recall@{1, 5, 10} |
| **Zero-shot segmentation** | ADE20K (150 classes), Puzzle-Perception (30 classes) | open-vocabulary mIoU (cosine sim between patch features and per-class text embeddings; no segmentation head trained) |

The headline metrics in `evaluation/results/*.csv` are the published
paper numbers, populated by [`scripts/ingest_vla_results.py`](../../scripts/ingest_vla_results.py)
from a hard-coded `PAPER_TABLE` dictionary. Live runs of `run_eval.py`
write under `evaluation/results/<task>/_live/` and never overwrite the
published values.

## Supported models

Three model tags configured in [`configs/models.yaml`](configs/models.yaml):

| Tag | Vision encoder | Text encoder | Checkpoint |
|---|---|---|---|
| `clip` | CLIP ViT-L/14 @ 336 | CLIP text transformer | HuggingFace `openai/clip-vit-large-patch14-336` |
| `tdn`  | DINOv3-H+ + 2 trained head blocks | RoBERTa-large + 2 trained head blocks | `vith_roberta_v3_coco_ft/ckpt/99` |
| `tddn` | DINOv3-H+ + CleanDIFT (SD-2.1) fused vision + 2 trained head blocks | RoBERTa-large + 2 trained head blocks | average of `fused_dinov3_cleandift_coco_ft/ckpt/{149, 199}` |

The trained encoder modules are imported from
[`experiments/shared_utils/feature_extraction/text_alignment/`](../shared_utils/feature_extraction/text_alignment/);
trained-head checkpoints live under
[`experiments/shared_utils/feature_extraction/checkpoints/`](../shared_utils/feature_extraction/checkpoints/).

## Setup

```bash
export HF_TOKEN=<your_token>     # gated DINOv3 / RoBERTa weights
```

Datasets are resolved relative to the repository root. The code expects
them at:

| Symbol | Path |
|---|---|
| `cifar100` | `datasets/Existing_Datasets/Classification/CIFAR-100/cifar100/` |
| `caltech101` | `datasets/Existing_Datasets/Classification/Caltech-101/caltech101/` |
| `food101` | `datasets/Existing_Datasets/Classification/Food-101/` |
| `gtsrb` | `datasets/Existing_Datasets/Classification/GTSRB/` |
| `imagenet1k` | `datasets/Existing_Datasets/Classification/ImageNet-1K/imagenet_hf/` |
| `flickr30k` | `datasets/Existing_Datasets/Retrieval/Flickr30K/flickr30k/` |
| `coco` (val + train) | `datasets/Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014/` |
| `ade20k` | `datasets/Existing_Datasets/Segmentation/ADE20K/ade20k/` |
| `puzzle` | `datasets/Puzzle_Perception/Segmentation/data/` (downloaded via [`datasets/download_datasets.py`](../../datasets/download_datasets.py)) |

## Evaluate

A single combination — model ∈ {`clip`, `tdn`, `tddn`}, dataset ∈
{`cifar100`, `caltech101`, `food101`, `gtsrb`, `imagenet1k`,
`flickr30k`, `coco`, `ade20k`, `puzzle`}:

```bash
python run_eval.py --task classification --model tdn --dataset cifar100                  # zero-shot templates
python run_eval.py --task classification --model tdn --dataset cifar100 --mode cupl      # zero-shot CuPL
python run_eval.py --task classification --model tdn --dataset cifar100 --mode tip --k 16  # few-shot TIP-Adapter
python run_eval.py --task retrieval     --model tdn --dataset flickr30k
python run_eval.py --task segmentation  --model tdn --dataset ade20k
```

Any of `--task`, `--model`, `--dataset`, `--mode` accepts `all`; the
cross-product is dispatched in one go and a summary table prints at
the end:

```bash
python run_eval.py --task classification --model all --dataset cifar100 --mode zero_shot
python run_eval.py --task all            --model tdn --dataset all
python run_eval.py --task all            --model all --dataset all --limit 200
```

`--limit N` caps the number of evaluation samples per combo for quick
checks. Every combo writes a JSON under
`evaluation/results/<task>/_live/<model>_<dataset>[_<mode>].json`; the
headline CSVs are not modified.

CuPL falls back to template prompts when a dataset's descriptions JSON
is missing (applies to ImageNet-1K, which doesn't ship descriptions in
this tree). The fallback is recorded in the live JSON as
`"used_template_fallback": true`.

## Ablations

The TIP-Adapter K ∈ {1, 2, 4, 8, 16} sweep is a single command via
`--k-sweep`:

```bash
python run_eval.py --task classification --model tdn --dataset caltech101 \
                   --mode tip --k-sweep 1,2,4,8,16
# Produces _live/tdn_caltech101_tip_k{1,2,4,8,16}.json
```

The paper-canonical aggregate is committed at
[`evaluation/results/classification/tip_k_sweep.csv`](evaluation/results/classification/tip_k_sweep.csv).

## Train

Two-round protocol; both rounds use FSDP2 with `bf16` parameters and
`fp32` gradient reduction, plus selective activation checkpointing and
`torch.compile` on the trainable head blocks:

| Round | Data | LR | Iters | Grad-cache micro-batches |
|---|---|---|---|---|
| 1 | LAION + COCO (`ConcatDataset`) | 1e-3 | 5000 | 16 |
| 2 | COCO only (fine-tune of round-1 final) | 1e-4 | 500 (TDN) / 200 (TDDN) | 32 |

Loss = symmetric InfoNCE with all-gathered negatives + JS-divergence
structure regularizer (λ=10, level=1) against the frozen-backbone
reference embeddings. Hyperparameters live in
[`configs/training/round_{1,2}.yaml`](configs/training/) and
[`configs/models.yaml`](configs/models.yaml). Edit the YAMLs to tune;
the runner is intentionally CLI-light.

Launch under `torchrun` (single- or multi-GPU; the script relies on a
process group for FSDP + DCP checkpointing):

```bash
torchrun --nproc_per_node=N python run_train.py --variant tdn  --round 1
torchrun --nproc_per_node=N python run_train.py --variant tddn --round 2 \
    --resume-checkpoint <path/to/round1/ckpt/4999>
```

`--nproc_per_node=1` is fine for a sanity check. A plain
`python run_train.py ...` invocation exits immediately with a launcher
hint.

**Effective contrastive batch scales with GPU count.** At each
iteration the loss sees `batch_size × grad_cache_multiplier ×
world_size` (image, caption) pairs:

| Round | 1 GPU | 4 GPUs (published) | 8 GPUs |
|---|---:|---:|---:|
| 1 (`batch=64`, `grad_cache=16`) | 1,024 | **4,096** | 8,192 |
| 2 (`batch=64`, `grad_cache=32`) | 2,048 | **8,192** | 16,384 |

The published checkpoints were trained at 4 GPUs. If you train on
fewer GPUs and want to keep the same effective batch (contrastive
quality is batch-size-sensitive), bump
`gradient_cache.grad_cache_multiplier` in
[`configs/training/round_{1,2}.yaml`](configs/training/)
proportionally — e.g. 1 GPU + round-1 with `grad_cache_multiplier=64`
reproduces the 4-GPU effective batch of 4,096.

Checkpoints land at
`<out_root>/checkpoints/<variant>-round<n>/ckpt/<step>/` as DCP shards
(`.metadata` + `__<rank>_0.distcp` per rank) — the same on-disk format
the inference loaders consume. The merged config used for the run is
saved alongside as `config.yaml`.

The bundled checkpoints under
`experiments/shared_utils/feature_extraction/checkpoints/{vith_roberta_v3_coco_ft,fused_dinov3_cleandift_coco_ft}/`
already provide the published round-2 finals; `run_train.py` is for
retraining on new data.

## Results

| File | Contents |
|---|---|
| [`evaluation/results/classification/classification.csv`](evaluation/results/classification/classification.csv) | headline (`model, dataset, mode, top1`) |
| [`evaluation/results/retrieval/retrieval.csv`](evaluation/results/retrieval/retrieval.csv) | headline (`model, dataset, i2t_r1, t2i_r1`) |
| [`evaluation/results/segmentation/segmentation.csv`](evaluation/results/segmentation/segmentation.csv) | headline (`model, dataset, miou`) |
| `evaluation/results/classification/tip_k_sweep.csv` | TIP-Adapter K ∈ {1, 2, 4, 8, 16} ablation |
| `evaluation/results/<task>/<model>_<dataset>[_<mode>].json` | paper-canonical per-run detail (top1 / per-class IoU / recall@K) |
| `evaluation/results/<task>/_live/<model>_<dataset>[_<mode>].json` | live runs of `run_eval.py`; not committed |

## Prompts

[`evaluation/prompts/`](evaluation/prompts/) holds the text inputs the
encoder consumes to build per-class classifiers:

- `<dataset>.py` — `*_CLASSES` list per dataset (CIFAR-100 / Caltech-101 /
  Food-101 / GTSRB / ImageNet-1K, ADE20K, Puzzle).
- `openai_templates.py` — canonical 80 OpenAI zero-shot templates
  (Radford et al., 2021), shared across all classification + zero-shot
  segmentation datasets.
- `descriptions/<dataset>.json` — CuPL LLM-generated descriptions for
  zero-shot CuPL classification, 50 descriptions per class. Ships for
  CIFAR-100, Caltech-101, Food-101, GTSRB. Missing JSONs fall back to
  the template ensemble at runtime.

## Files

```
Vision_Language_Alignment/
├── README.md                  (this file)
├── configs/
│   ├── models.yaml            # per-variant arch + round overrides
│   └── training/
│       ├── round_1.yaml       # LAION + COCO pretrain hyperparameters
│       └── round_2.yaml       # COCO fine-tune hyperparameters
├── evaluation/
│   ├── prompts/               # class names + templates + CuPL descriptions
│   ├── results/               # paper-canonical CSVs + detail JSONs (+ _live/ from local runs)
│   └── src/                   # classifier / retrieval / segmentation / TIP-Adapter / encoder adapters / dataset wrappers
├── run_eval.py                # eval entry point (per-combo or sweep)
├── run_train.py               # training entry point (launch under torchrun)
└── training/
    └── src/                   # data loaders, losses, FSDP setup, checkpoint I/O, training loop
```
