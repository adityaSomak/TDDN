# DiffusedDINOv1

A study of **vision encoders that fuse self-supervised and diffusion
features**. The central hypothesis: by combining the dense semantic
structure of DINOv3 with the spatial / texture cues that fall out of
Stable Diffusion's UNet (via the CleanDIFT reformulation), we can
build vision encoders that beat either source alone on *both*
recognition (k-NN, classification) and localization (segmentation,
keypoint matching) — and that also align cleanly with a frozen
RoBERTa text encoder.

The repo evaluates three model families across one shared registry:

- **Baselines** — DINOv3, DINOv2, CLIP, Stable Diffusion 2.1, CleanDIFT.
- **Handcraft fusion** — `ddn`: L2-normalized concatenation of frozen
  DINOv3 patches with CleanDIFT layer features (no training).
- **Trained alignment** — `tdn` (DINOv3-H+ vision + RoBERTa-large text
  + two trained head blocks per encoder) and `tddn` (the same trained
  heads on top of a fused DINOv3 + CleanDIFT vision encoder). Both are
  trained with a CLIP-style symmetric InfoNCE loss plus a
  structure-preservation regularizer (Jensen-Shannon on softmaxed
  similarity matrices) on LAION + COCO captions.

Six independent evaluation tracks under [`experiments/`](experiments/)
measure these encoders against complementary criteria:

| Track | Tests | Headline metric |
|---|---|---|
| [Representation_Analysis](experiments/Representation_Analysis/) | intrinsic feature quality + cross-encoder similarity | CKA / PWCCA + uniformity / effective-rank |
| [Segmentation](experiments/Segmentation/) | linear-probe dense prediction (frozen backbone, trained head) | weighted mIoU on Puzzle-Perception (30 classes) |
| [Keypoint_Matching](experiments/Keypoint_Matching/) | fine-grained spatial correspondence | PCK@{0.1, 0.05, 0.01} on SPair-71K |
| [imagenet_knn](experiments/imagenet_knn/) | global-feature semantic separability | top-1 / top-5 k-NN (k=20) on ImageNet-1K |
| [Vision_Language_Alignment](experiments/Vision_Language_Alignment/) | end-to-end vision-language usefulness | top-1 (zero-shot / CuPL / TIP-Adapter) + Recall@1 + zero-shot open-vocab mIoU |
| [Puzzle_Understanding](experiments/Puzzle_Understanding/) | do TDDN segmentation masks help downstream VLMs reason about algorithmic puzzles? | per-task accuracy across GPT-5.x / Qwen2.5-VL / InternVL3 / ... |

Each `experiments/<name>/README.md` documents how to run its
evaluation; this top-level README handles shared setup and conventions.

## Repository layout

```
DiffusedDINOv1/
├── README.md                              # this file
├── requirements.txt                       # main env (every experiment)
├── requirements_vllm.txt                  # separate env for vLLM-served VLMs
├── datasets/                              # shared dataset trees
├── scripts/                               # repo-level helpers (e.g. ingest_*.py)
└── experiments/
    ├── shared_utils/
    │   └── feature_extraction/            # unified extractor + transform registry
    │
    ├── Representation_Analysis/           # CKA + PWCCA similarity, uniformity / effective-rank quality + PCA→RGB activation maps
    │   ├── README.md
    │   ├── run.py                         # CLI: plots / activation-maps / metrics
    │   ├── configs/                       # metrics + models + activation-maps + coco sample IDs
    │   ├── metrics/                       # similarity / quality / feature_utils + third_party CCA
    │   ├── pca_viz/                       # render_one() — single-image activation map
    │   ├── qualitative/                   # input samples + per-model activation-map outputs
    │   │   ├── README.md
    │   │   ├── samples/
    │   │   └── {baselines,ddn,tdn,tddn}/
    │   └── quantitative/                  # CSV + plot outputs
    │       ├── README.md
    │       ├── global/{results,plots}/
    │       └── patch/{results,plots}/
    │
    ├── Segmentation/                      # linear-probe segmentation on Puzzle-Perception (30-class unified label space). Headline: weighted mIoU.
    │   ├── README.md
    │   ├── run_train.py
    │   ├── run_eval.py
    │   ├── configs/                       # models + training
    │   ├── training/{src,checkpoints,logs}/
    │   └── evaluation/{src,results}/
    │
    ├── Keypoint_Matching/                 # SPair-71K PCK@α evaluation. Headline: PCK@0.1 (bbox).
    │   ├── README.md
    │   ├── run_eval.py
    │   ├── configs/                       # models.yaml (11 tags)
    │   └── evaluation/{src,results,results/ablations}/
    │
    ├── imagenet_knn/                      # k-NN (k=20) classification on ImageNet-1K from image-encoder features. Headline: top-1 accuracy.
    │   ├── README.md
    │   ├── run_eval.py
    │   ├── configs/                       # models.yaml (11 tags)
    │   └── evaluation/{src,results}/
    │
    ├── Vision_Language_Alignment/         # zero-shot / few-shot classification + bidirectional retrieval + zero-shot open-vocab segmentation across {clip, tdn, tddn}.
    │   ├── README.md
    │   ├── run_eval.py
    │   ├── run_train.py
    │   ├── configs/                       # models.yaml + training/{round_1,round_2}.yaml
    │   ├── evaluation/                    # prompts (class names + 80 OpenAI templates + CuPL JSONs) + src/ (classifier, retrieval, segmentation, TIP-Adapter) + results/ (paper-canonical CSVs + per-run JSONs)
    │   └── training/                      # data, losses, FSDP setup, checkpoint I/O, training loop
    │
    └── Puzzle_Understanding/              # VLM evaluation on AlgoPuzzleVQA tasks (full Q1..QN + seg-eval grid reconstruction)
        ├── README.md
        ├── run_full_eval.py
        ├── run_seg_eval.py
        ├── prompts/                       # per-task prompt registries
        ├── scripts/                       # launch_vllm.sh for open-source VLMs
        └── results/{full_eval,seg_eval}/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e <path>/dinov3            # Meta AI DINOv3 reference impl
```

For the open-source VLM serving used by
`Puzzle_Understanding/scripts/launch_vllm.sh`, install
[`requirements_vllm.txt`](requirements_vllm.txt) into a **separate**
venv (vLLM has a heavy CUDA dependency tree that shouldn't pollute the
main env). See
[`experiments/Puzzle_Understanding/README.md`](experiments/Puzzle_Understanding/README.md)
for the two-env install recipe.

## Environment variables

A template `.env` lives at the repo root (gitignored). Fill in your
values and export them to the current shell:

```bash
# edit .env — set HF_TOKEN (and OPENAI_API_KEY if you'll use OpenAI VLMs)
set -a; source .env; set +a
```

`.env` is not auto-loaded — every script reads from `os.environ`
directly. The variables it can set:

| Variable | Required for | Default |
|---|---|---|
| `HF_TOKEN` | every experiment that loads DINOv3 or RoBERTa-large (i.e. all but `Puzzle_Understanding`) | — |
| `OPENAI_API_KEY` | `Puzzle_Understanding/run_*_eval.py --backend openai` | — |
| `EXPERIMENTS_DATASETS_ROOT` | optional override of [`shared_utils.paths.DATASETS_ROOT`](experiments/shared_utils/paths.py) | repo-root `datasets/` |
| `EXPERIMENTS_CHECKPOINTS_ROOT` | optional override of `CHECKPOINTS_ROOT` | `experiments/shared_utils/feature_extraction/checkpoints/` |
| `EXPERIMENTS_FEATURES_ROOT` | optional override of `FEATURES_ROOT` (where the `Representation_Analysis` metrics CLI caches per-image extracted features) | repo-root `.features_cache/` |
| `VLLM_PY` | optional path to the vLLM venv's Python interpreter — only used by `Puzzle_Understanding/scripts/launch_vllm.sh` | first `python` on `PATH` |

## Shared assets

- **Datasets** live under [`datasets/`](datasets/) by default; redirect
  via `EXPERIMENTS_DATASETS_ROOT` to keep large trees outside the repo.
- **Backbone checkpoints** live under
  [`experiments/shared_utils/feature_extraction/checkpoints/`](experiments/shared_utils/feature_extraction/);
  per-experiment READMEs list which ones each experiment needs.

## Per-experiment entry points

- [`experiments/Representation_Analysis/`](experiments/Representation_Analysis/README.md)
- [`experiments/Segmentation/`](experiments/Segmentation/README.md)
- [`experiments/Keypoint_Matching/`](experiments/Keypoint_Matching/README.md)
- [`experiments/imagenet_knn/`](experiments/imagenet_knn/README.md)
- [`experiments/Vision_Language_Alignment/`](experiments/Vision_Language_Alignment/README.md)
- [`experiments/Puzzle_Understanding/`](experiments/Puzzle_Understanding/README.md)
- [`experiments/shared_utils/feature_extraction/`](experiments/shared_utils/feature_extraction/README.md)
