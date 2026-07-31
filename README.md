# DiffusedDINOv1

Solving structured reasoning tasks over images (such as image puzzle
solving) requires reasoning over fine-grained visual perception
abilities — which seems missing from current state-of-the-art
Vision-Language Models (VLMs). In fact, VLMs using CLIP-based ViT
backbones seem to sacrifice fine-grained visual details to capture
high-level semantic understanding. We empirically observe that
failure of these ViT backbones propagates to downstream tasks when
attempting to solve visual puzzles. Therefore, we propose a novel
visual perception module (**DiffusedDINO**) that integrates ViT
(DINOv3) and Diffusion (CleanDIFT) representations to achieve a
superior fine-coarse tradeoff, suitable for downstream visual
reasoning tasks. We perform detailed analysis of the effectiveness
of this fused representation. Representation-quality analysis along
with performance on image perception tasks show informativeness and
discriminative properties of this representation. We specifically
benchmark effectiveness in the semantic-segmentation task using a
novel puzzle-based segmentation dataset (namely **Puzzle
Perception**). Next, we show that traditional image-text alignment
techniques are sufficient to maintain effectiveness in multimodal
(vision-language) tasks such as image-text retrieval. Lastly, we
show that segmentation masks predicted using **DiffusedDINO** can
assist various VLMs to detect objects of various abstract shapes and
sizes, often with considerable improvements.

## What's in this repository

The paper's three model tags map to the codebase as follows:

- **`ddn` = DiffusedDINO** — the proposed visual perception module.
  L2-normalized concatenation of frozen DINOv3 patches with frozen
  CleanDIFT layer features; no training.
- **`tdn` = text-aligned DINOv3** — DINOv3-H+ vision + RoBERTa-large
  text + two trained alignment-head blocks per encoder. Baseline
  showing how pure-DINOv3 alignment performs without the diffusion
  fusion.
- **`tddn` = text-aligned DiffusedDINO** — the same trained alignment
  heads on top of the `ddn` (DiffusedDINO) fused vision encoder. The
  vision-language-aligned version of DiffusedDINO.

Both `tdn` and `tddn` are trained with a CLIP-style symmetric
InfoNCE loss plus a structure-preservation regularizer (Jensen-
Shannon on softmaxed similarity matrices) on LAION + COCO captions.
The remaining tags (`dinov3`, `dinov2-vitb`, `dinov2-vitg`, `clip`,
`sd-2.1`, `cd`, `sd+dinov2-vitb`, `sd+dinov2-vitg`) are unmodified
baselines wired through the same registry.

Seven independent evaluation tracks under [`experiments/`](experiments/)
benchmark these encoders against complementary criteria:

| Track | Tests | Headline metric |
|---|---|---|
| [Representation_Analysis](experiments/Representation_Analysis/) | intrinsic feature quality + cross-encoder similarity | CKA / PWCCA + uniformity / effective-rank |
| [Segmentation](experiments/Segmentation/) | linear-probe dense prediction (frozen backbone, trained head) | weighted mIoU on Puzzle-Perception (30 classes) |
| [Keypoint_Matching](experiments/Keypoint_Matching/) | fine-grained spatial correspondence | PCK@{0.1, 0.05, 0.01} on SPair-71K |
| [ImageNet_Classification](experiments/ImageNet_Classification/) | global-feature semantic separability | top-1 / top-5 k-NN (k=20) on ImageNet-1K |
| [Vision_Language_Alignment](experiments/Vision_Language_Alignment/) | end-to-end vision-language alignment, benchmarked against 5 public CLIP-lineage baselines | top-1 (zero-shot / CuPL / TIP-Adapter) + Recall@1 + zero-shot open-vocab mIoU |
| [Puzzle_Understanding](experiments/Puzzle_Understanding/) | do TDDN segmentation masks help downstream VLMs reason about algorithmic puzzles? *(earlier track, kept as originally written)* | per-task accuracy across GPT-5.x / Qwen2.5-VL / InternVL3 / ... |
| [CRG](experiments/CRG/) | can TDDN-predicted regions replace GT regions in a decode-side intervention on a *frozen* VLM? | Δ question accuracy vs. the raw image, macro over questions (8 VLMs × 2 puzzles) |

Each `experiments/<name>/README.md` documents how to run its
evaluation; this top-level README handles shared setup and conventions.

Most tracks share one layout — `run_eval.py` at the root over
`configs/models.yaml` and `evaluation/{src,results}/`. `Puzzle_Understanding`
predates that convention and is deliberately left in its original form.

## Repository layout

```
DiffusedDINOv1/
├── README.md                              # this file
├── requirements.txt                       # single env for every experiment
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
    ├── ImageNet_Classification/                      # k-NN (k=20) classification on ImageNet-1K from image-encoder features. Headline: top-1 accuracy.
    │   ├── README.md
    │   ├── run_eval.py
    │   ├── configs/                       # models.yaml (11 tags)
    │   └── evaluation/{src,results}/
    │
    ├── Vision_Language_Alignment/         # zero-shot / few-shot classification + bidirectional retrieval + sliding-window open-vocab segmentation, across 8 models (clip, tdn, tddn + 5 public CLIP-lineage baselines).
    │   ├── README.md
    │   ├── run_eval.py
    │   ├── run_train.py
    │   ├── configs/                       # models.yaml + training/{round_1,round_2}.yaml
    │   ├── evaluation/                    # prompts (class names + 80 OpenAI templates + CuPL JSONs) + src/ (classifier, retrieval, segmentation, TIP-Adapter) + results/ (paper-canonical CSVs + per-run JSONs)
    │   └── training/                      # data, losses, FSDP setup, checkpoint I/O, training loop
    │
    ├── Puzzle_Understanding/              # VLM evaluation on AlgoPuzzleVQA tasks (full Q1..QN + seg-eval grid reconstruction). Earlier track: keeps its own prompts/ + results/ layout rather than the configs/ + evaluation/ one the others use.
    │   ├── README.md
    │   ├── run_full_eval.py
    │   ├── run_seg_eval.py
    │   ├── prompts/                       # per-task prompt registries
    │   ├── scripts/                       # launch_vllm.sh for open-source VLMs
    │   └── results/{full_eval,seg_eval}/
    │
    └── CRG/                               # Contrastive Region Guidance: do TDDN-predicted regions improve a frozen VLM's perception at decode time?
        ├── README.md
        ├── run_eval.py                    # eval + --aggregate / --validate-* / --redetect
        ├── run_generate.py                # mint a NEW chess board set (rare, destructive)
        ├── configs/models.yaml            # the 8 probed VLMs + defaults (alpha, arms, TDDN tuning)
        └── evaluation/                    # src/ (data, negatives, CRG decode, per-task evals, aggregate, TDDN) + results/ (paper table + per-model per-question JSONs + qualitative figures)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e <path>/dinov3            # Meta AI DINOv3 reference impl
```

For the open-source VLM serving used by
`Puzzle_Understanding/scripts/launch_vllm.sh`, install `vllm` into a
**separate** venv (it has a heavy CUDA dependency tree that shouldn't
pollute the main env). See
[`experiments/Puzzle_Understanding/README.md`](experiments/Puzzle_Understanding/README.md)
for that install recipe.

## Environment variables

Create a `.env` at the repo root with the variables below, then export
them to the current shell:

```bash
set -a; source .env; set +a
```

`.env` is gitignored and not auto-loaded — every script reads from
`os.environ` directly.

**Required**
- `HF_TOKEN` — gated DINOv3 / RoBERTa-large weights; needed for every
  experiment except `Puzzle_Understanding`.
- `OPENAI_API_KEY` — only for `Puzzle_Understanding/run_*_eval.py
  --backend openai`.

**Optional**
- `VLLM_PY` — path to the vLLM venv's Python interpreter, used by
  `Puzzle_Understanding/scripts/launch_vllm.sh`. Defaults to the
  first `python` on `PATH`.
- `DINOV3_ROOT` — path to the Meta DINOv3 source tree. Needed only when
  `dinov3` is not `pip install -e`'d: by the `mask_generation` overlays
  and by `CRG/run_eval.py --redetect`.

If you need to keep datasets / checkpoints / the metrics feature
cache outside the repo tree, four `EXPERIMENTS_*_ROOT` overrides
exist — including `EXPERIMENTS_LOCAL_DATA_ROOT` for the
locally-supplied data in [`datasets/_local/`](datasets/_local/README.md).
See [`shared_utils/paths.py`](experiments/shared_utils/paths.py).

## Per-experiment entry points

- [`experiments/Representation_Analysis/`](experiments/Representation_Analysis/README.md)
- [`experiments/Segmentation/`](experiments/Segmentation/README.md)
- [`experiments/Keypoint_Matching/`](experiments/Keypoint_Matching/README.md)
- [`experiments/ImageNet_Classification/`](experiments/ImageNet_Classification/README.md)
- [`experiments/Vision_Language_Alignment/`](experiments/Vision_Language_Alignment/README.md)
- [`experiments/Puzzle_Understanding/`](experiments/Puzzle_Understanding/README.md)
- [`experiments/CRG/`](experiments/CRG/README.md)
- [`experiments/shared_utils/feature_extraction/`](experiments/shared_utils/feature_extraction/README.md)
