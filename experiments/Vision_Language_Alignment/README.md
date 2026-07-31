# Vision_Language_Alignment

Evaluates image–text alignment encoders on three tasks, against eight models on
equal footing. Every number comes from one entry point over one model registry.

| Task | Datasets | Metric |
|---|---|---|
| Classification | CIFAR-100, Caltech-101, Food-101, GTSRB | top-1 (zero-shot / CuPL / TIP-Adapter) |
| Retrieval | Flickr30K, COCO val2014 | bidirectional Recall@{1,5,10} |
| Segmentation | ADE20K, Cityscapes, COCO-Stuff, PASCAL-Context-59, Puzzle-Perception | zero-shot open-vocab mIoU |

Segmentation follows the GroupViT/TCL sliding-window protocol: resize the shorter
side to the dataset's target (aspect preserved, long side capped at 2048), cover
the image with `crop`-sized windows at half-crop stride, average the overlaps, and
score at that resolution — no fixed square, no letterbox. ADE20K, Cityscapes,
COCO-Stuff and PASCAL-Context use a 448 target; Puzzle-Perception uses 336.

**Retrieval candidate sets differ, and the two datasets' numbers are not on a
common scale.** Flickr30K uses the 1,000-image Karpathy test split — the published
protocol. COCO scores against all 40,504 val2014 images rather than the
5,000-image Karpathy test split, so its Recall@K sits well below figures reported
elsewhere. The candidate set is recorded as `protocol` in every retrieval result
and as a column in `retrieval.csv`: comparisons between models are valid,
comparisons of the COCO column against published numbers are not.

## Supported models

Eight tags in [`configs/models.yaml`](configs/models.yaml), which is the single
source of truth — `evaluation/src/encoders.py` builds both the global and the dense
encoder for a tag from its entry there.

| Tag | Vision encoder | Text encoder | Checkpoint |
|---|---|---|---|
| `clip` | CLIP ViT-L/14 @ 336 | CLIP | `openai/clip-vit-large-patch14-336` |
| `metaclip_l14` | ViT-L/14 (QuickGELU) | MetaCLIP | `metaclip_fullcc` |
| `dfn_l14` | ViT-L/14 (QuickGELU) | DFN | `dfn2b` |
| `openclip_l14` | ViT-L/14 | OpenCLIP | `laion2b_s32b_b82k` |
| `siglip2_l16` | ViT-L/16 SigLIP2 @ 384 | SigLIP2 | `webli` |
| `fgclip2_large` | FG-CLIP2 large | FG-CLIP2 | `qihoo360/fg-clip2-large` |
| `tdn` | DINOv3 ViT-H+ + 2 trained head blocks | RoBERTa-large + 2 trained head blocks | `vith_roberta_v3_coco_ft/ckpt/tdn` |
| `tddn` | DINOv3 + CleanDIFT (fused) + 2 trained head blocks | RoBERTa-large + 2 trained head blocks | `fused_dinov3_cleandift_coco_ft/ckpt/tddn` |

## Setup

```bash
export HF_TOKEN=<your_token>          # gated DINOv3 / RoBERTa weights
pip install -r ../../requirements.txt
```

Trained-head tags need their checkpoints under
`experiments/shared_utils/feature_extraction/checkpoints/`:

- `tdn` → `vith_roberta_v3_coco_ft/ckpt/tdn/`
- `tddn` → `fused_dinov3_cleandift_coco_ft/ckpt/tddn/`

Datasets, all repo-relative:

| Symbol | Path |
|---|---|
| `cifar100` | `datasets/Existing_Datasets/Classification/CIFAR-100` |
| `caltech101` | `datasets/Existing_Datasets/Classification/Caltech-101` |
| `food101` | `datasets/Existing_Datasets/Classification/Food-101` |
| `gtsrb` | `datasets/Existing_Datasets/Classification/GTSRB` |
| `flickr30k` | `datasets/Existing_Datasets/Retrieval/Flickr30K` |
| `coco` | `datasets/Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014/val2014` |
| `ade20k` | `datasets/Existing_Datasets/Segmentation/ADE20K` |
| `cityscapes` | `datasets/Existing_Datasets/Segmentation/Cityscapes` |
| `coco_stuff` | `datasets/Existing_Datasets/Segmentation/COCO_Stuff` |
| `context59` | `datasets/Existing_Datasets/Segmentation/PASCAL_Context` |
| `puzzle_perception` | `datasets/Puzzle_Perception/Segmentation/data` |

Fetch the public ones with `python datasets/download_datasets.py --dataset <name>`.

## Evaluate

```bash
python run_eval.py --task classification --model tddn --dataset cifar100
python run_eval.py --task classification --model tddn --dataset cifar100 --mode cupl
python run_eval.py --task classification --model tddn --dataset cifar100 --mode tip --k 16
python run_eval.py --task retrieval      --model clip --dataset flickr30k
python run_eval.py --task segmentation   --model tddn --dataset cityscapes
python run_eval.py --task segmentation   --model tddn --dataset ade20k --limit 8   # smoke test
```

`--task`, `--model`, `--dataset` and `--mode` each accept `all`; the cross-product
is dispatched in one go and a summary table prints at the end:

```bash
python run_eval.py --task all --model all --dataset all --limit 200
```

| Arg | Default | Meaning |
|---|---|---|
| `--task` | _required_ (except `--aggregate`) | `classification` / `retrieval` / `segmentation` / `all` |
| `--model` | _required_ | a registry tag, or `all` |
| `--dataset` | _required_ | a dataset name, or `all` to expand within the task |
| `--mode` | `zero_shot` | classification only: `zero_shot` / `cupl` / `tip` / `all` |
| `--k` | `16` | TIP-Adapter shots per class |
| `--k-sweep` | — | comma-separated shot counts, e.g. `1,2,4,8,16` |
| `--limit` | all | cap samples per combo |
| `--max-batch` | `32` | segmentation: max windows per forward |
| `--images-per-chunk` | `32` | segmentation: images accumulated before scoring |
| `--force` | off | recompute even if the output already exists |
| `--publish` | off | write the committed result file instead of `_live/` |
| `--aggregate` | — | rebuild the headline CSVs from committed results and exit |
| `--device` | `cuda` if available, else `cpu` | |

Runs write to `evaluation/results/<task>/_live/` unless `--publish` is given, so a
smoke test cannot overwrite a published number. `_live/` is not committed.

CuPL falls back to template prompts when `descriptions/<dataset>.json` is missing;
the fallback is recorded as `"used_template_fallback": true`.

## Ablations

TIP-Adapter shot count, one command per (model, dataset):

```bash
python run_eval.py --task classification --model tddn --dataset caltech101 \
    --mode tip --k-sweep 1,2,4,8,16
```

Writes `_live/tddn_caltech101_tip_k{1,2,4,8,16}.json`; `--aggregate` folds them
into `tip_k_sweep.csv`.

## Results

Committed artifacts. The three main CSVs are regenerated from the per-model JSONs
by `--aggregate`, so they carry nothing the JSONs don't:

- [`evaluation/results/classification/classification.csv`](evaluation/results/classification/classification.csv)
  — `model, dataset, mode, top1`. 48 rows: 8 models × 4 datasets zero-shot, plus
  `tdn`/`tddn` CuPL and TIP-Adapter.
- [`evaluation/results/retrieval/retrieval.csv`](evaluation/results/retrieval/retrieval.csv)
  — `model, dataset, protocol, n_images, i2t_r1, t2i_r1`. 16 rows.
- [`evaluation/results/segmentation/segmentation.csv`](evaluation/results/segmentation/segmentation.csv)
  — `model, dataset, miou` (percent). 40 rows: 8 models × 5 datasets.
- [`evaluation/results/classification/tip_k_sweep.csv`](evaluation/results/classification/tip_k_sweep.csv)
  — `model, dataset, k_1, k_2, k_4, k_8, k_16`, 8 rows. Retained from an earlier
  sweep whose per-run JSONs are not in the tree, so unlike the three above it is
  not currently regenerable; `--aggregate` leaves it untouched. Re-run
  `--mode tip --k-sweep 1,2,4,8,16` per (model, dataset) to rebuild it.

Per-run detail, `<model>_<dataset>[_<mode>].json` under each task directory:

| Task | Keys |
|---|---|
| classification | `model, dataset, mode, n_samples, top1, top5`; TIP instead carries `k, n_cache, n_query, per_alpha, best_alpha, best_top1` |
| retrieval | `model, dataset, protocol, n_images, n_captions, i2t_r{1,5,10}, t2i_r{1,5,10}` |
| segmentation | `model, dataset, protocol, n_classes, n_images, miou, per_class_iou` |

mIoU is a percent in every committed artifact. The `cupl` and `tip` files carry
`model, dataset, mode, top1` only — they predate the current writer, which also
records sample counts. Likewise, only the `puzzle_perception` segmentation
files carry `per_class_iou`; the other four datasets' files predate the writer
adding it. Re-run to backfill (`--force --publish`).

## Prompts

- `evaluation/prompts/<dataset>.py` — one ALL-CAPS class-name list per dataset,
  ordered to match the dataset's integer labels. `context59` reads
  `pascal_context.py`; every other dataset key matches its module name.
- `evaluation/prompts/openai_templates.py` — the 80 OpenAI prompt templates, as
  callables.
- `evaluation/prompts/descriptions/<dataset>.json` — 50 CuPL descriptions per class.

## Train

Two rounds, configured in [`configs/training/`](configs/training/):

| Round | Data | LR | Iters | Grad-cache micro-batches |
|---|---|---|---|---|
| 1 | Recaptioned LAION + COCO | `1e-3` | 5,000 | 16 |
| 2 | COCO only | `1e-4` | 200–500 (per tag) | 32 |

```bash
torchrun --nproc_per_node=4 run_train.py --variant tddn --round 1
torchrun --nproc_per_node=4 run_train.py --variant tddn --round 2
```

The effective contrastive batch scales with GPU count; the published runs used 4.

## Files

```
Vision_Language_Alignment/
├── README.md                     (this file)
├── run_eval.py                   one entry point; --task / --model / --dataset
├── run_train.py
├── configs/
│   ├── models.yaml               8 tags — the single source of encoder wiring
│   └── training/{round_1,round_2}.yaml
├── evaluation/
│   ├── prompts/                  per-dataset class names + templates + CuPL
│   ├── src/
│   │   ├── encoders.py           build_alignment_encoder / build_dense_encoder
│   │   ├── datasets.py           per-task dataset builders
│   │   ├── classifier.py         text class-prototype matrices
│   │   ├── tip_adapter.py        few-shot cache + alpha sweep
│   │   ├── retrieval.py          bidirectional Recall@K
│   │   ├── segmentation.py       mIoU + running confusion
│   │   ├── slide_inference.py    sliding-window tiling + overlap averaging
│   │   └── aggregate.py          rebuilds the headline CSVs
│   └── results/{classification,retrieval,segmentation}/
└── training/src/                 data, losses, FSDP, checkpoint I/O, loop
```
