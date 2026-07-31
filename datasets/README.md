# Datasets

A consolidated dataset tree used by the experiments in this repository. It
groups public vision/language benchmarks under `Existing_Datasets/` and
custom puzzle-perception data under `Puzzle_Perception/`. Public benchmarks
are not redistributed — a helper script downloads them from their original
upstream source on demand.

## Layout

```
datasets/
├── download_datasets.py
├── Existing_Datasets/
│   ├── Classification/
│   │   ├── CIFAR-100/
│   │   ├── Caltech-101/
│   │   ├── Food-101/
│   │   ├── GTSRB/
│   │   └── ImageNet-1K/
│   ├── Keypoint_Matching/
│   │   └── SPair-71K/
│   ├── Retrieval/
│   │   └── Flickr30K/
│   ├── Segmentation/
│   │   ├── ADE20K/
│   │   ├── PASCAL_VOC/
│   │   ├── Cityscapes/
│   │   ├── COCO_Stuff/
│   │   └── PASCAL_Context/
│   ├── Vision_Language_Alignment/
│   │   ├── Recaptioned_LAION/
│   │   └── MS-COCO-2014/
│   └── PVQA/
│       ├── AlgoPuzzleVQA/
│       └── AlgoPuzzleVQA_star/
├── Puzzle_Perception/
│   ├── Segmentation/                 # combined 30-class seg dataset (HF download)
│   └── PVQA/{chess,nqueens}/         # CRG perception probes (900 boards, 1200 QA rows)
└── _local/                           # locally-supplied data; gitignored (see its README)
```

## Requirements

The repo-wide [`requirements.txt`](../requirements.txt) at the project
root supplies every Python dependency this script needs (`torch`,
`torchvision`, `datasets`, `huggingface_hub`, `pycocotools`, …), except
the `kaggle` CLI (`pip install kaggle`), needed only for `cityscapes`.
The only other system tool that must be on `PATH` is `curl` (used for
direct HTTP fetches of COCO / ADE20K / PASCAL / SPair); archive
extraction goes through the Python stdlib (`zipfile`, `tarfile`), so
`unzip` / `tar` are not required.

## Downloads

`download_datasets.py` is the single entry point for every dataset used by
this project. It fetches each dataset into its canonical subdirectory and
never writes anywhere outside `datasets/`.

```bash
# All datasets
python download_datasets.py --all

# One at a time
python download_datasets.py --dataset cifar100
python download_datasets.py --dataset puzzle_perception

# Overwrite an existing copy
python download_datasets.py --dataset cifar100 --force
```

| Dataset            | Target subdirectory                                  | Source                                                                                       |
|--------------------|------------------------------------------------------|----------------------------------------------------------------------------------------------|
| CIFAR-100          | `Existing_Datasets/Classification/CIFAR-100/`        | torchvision                                                                                  |
| Caltech-101        | `Existing_Datasets/Classification/Caltech-101/`      | [`HuggingFaceM4/Caltech-101`](https://huggingface.co/datasets/HuggingFaceM4/Caltech-101) (HF mirror; torchvision's built-in Google Drive link is dead) |
| Food-101           | `Existing_Datasets/Classification/Food-101/`         | torchvision                                                                                  |
| GTSRB              | `Existing_Datasets/Classification/GTSRB/`            | torchvision                                                                                  |
| ImageNet-1K        | `Existing_Datasets/Classification/ImageNet-1K/imagenet_hf/` | [`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k) (gated)     |
| SPair-71K          | `Existing_Datasets/Keypoint_Matching/SPair-71K/`     | [postech.ac.kr](https://cvlab.postech.ac.kr/research/SPair-71k/)                             |
| Flickr30K          | `Existing_Datasets/Retrieval/Flickr30K/`             | [`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k)                     |
| ADE20K             | `Existing_Datasets/Segmentation/ADE20K/`             | [`ranksu/ADE20K`](https://huggingface.co/datasets/ranksu/ADE20K) (HF mirror of the official archive) |
| PASCAL VOC 2012    | `Existing_Datasets/Segmentation/PASCAL_VOC/`         | [pjreddie.com](https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar) (fast mirror of the official host) |
| Cityscapes         | `Existing_Datasets/Segmentation/Cityscapes/`         | [`kavithak1388/cityscapes`](https://www.kaggle.com/datasets/kavithak1388/cityscapes) (Kaggle mirror, needs a Kaggle token) |
| COCO-Stuff         | `Existing_Datasets/Segmentation/COCO_Stuff/`         | [images.cocodataset.org](http://images.cocodataset.org/) + [calvin.inf.ed.ac.uk](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/) (val2017 only) |
| PASCAL-Context59   | `Existing_Datasets/Segmentation/PASCAL_Context/`     | [host.robots.ox.ac.uk](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/) (VOC2010 images) + Wayback Machine snapshot of `trainval_merged.json` (Context annotations) |
| MS-COCO-2014       | `Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014/` | [images.cocodataset.org](http://images.cocodataset.org/)                              |
| Recaptioned LAION  | `Existing_Datasets/Vision_Language_Alignment/Recaptioned_LAION/data/` | [`PuzzleBench/Recaptioned_LAION`](https://huggingface.co/datasets/PuzzleBench/Recaptioned_LAION) (508k samples, ~72 GB; the downloader extracts 30 webdataset shards into a flat `images/` dir + filename-keyed `metadata.csv`) |
| Puzzle-Perception  | `Puzzle_Perception/Segmentation/data/`               | [`PuzzleBench/Puzzle_Perception`](https://huggingface.co/datasets/PuzzleBench/Puzzle_Perception) (combined 30-class segmentation) |

### Authentication

Gated datasets (ImageNet-1K and the `PuzzleBench/*` HuggingFace repos while
they remain private) require a HuggingFace access token:

```bash
export HF_TOKEN=hf_...
```

Request access at the dataset page before downloading.

`cityscapes` pulls from a Kaggle-hosted mirror and needs a Kaggle API
token at `~/.kaggle/kaggle.json` (create one under
[kaggle.com/settings](https://www.kaggle.com/settings) -> *Create New
Token*).

## PVQA

This repository ships the PVQA splits used in our evaluations,
committed under `Existing_Datasets/PVQA/`.

### `AlgoPuzzleVQA/`

Four algorithmic puzzle tasks — `checker_move`, `maze_solve`,
`nqueens`, `wood_slide` — each shaped as:

```
<task>/
├── <task>.csv              # puzzle metadata + start state
├── <task>_eval.jsonl       # evaluation questions
├── questions.yaml          # question templates
├── generate_*.py           # data-generation scripts
└── images/<NNNN>/
    ├── description.txt
    ├── plan_NLD.txt
    ├── planner_plan.txt
    ├── problem.pddl
    └── *.jpg               # rendered puzzle images
```

### `AlgoPuzzleVQA_star/`

A 100-puzzle evaluation subset for `maze` and `nqueens`, paired with the
segmentation overlays used in our experiments (oracle masks and predicted
masks).

```
<task>/
├── data/
│   ├── <task>_v2.csv             # puzzle metadata + Q/A annotations
│   ├── <task>_eval.jsonl         # evaluation questions
│   ├── <task>_eval_masked.jsonl  # mask-guided evaluation questions (nqueens)
│   ├── questions.yaml
│   └── images/<ID>/              # 100 puzzles per task
│       ├── description.txt, plan_NLD.txt, planner_plan.txt, problem.pddl
│       └── *.jpg
├── seg_data/
│   ├── oracle_mask/<ID>_overlay.jpg   # ground-truth region overlays
│   └── tddn_mask/<ID>_overlay.jpg     # predicted-mask overlays
└── scripts/
    └── generate_<task>_answers.py     # rebuilds <task>_v2.csv from upstream
```

The 100-ID set is identical across `maze`, `nqueens`, and `checker_move` and
is the binding subset used in all VLM evaluations.

A sibling `mask_generation/` folder ships the scripts that produced the
`seg_data/oracle_mask/` and `seg_data/tddn_mask/` overlays. See
[`AlgoPuzzleVQA_star/mask_generation/README.md`](Existing_Datasets/PVQA/AlgoPuzzleVQA_star/mask_generation/README.md)
for usage.

## Puzzle_Perception

Custom data published with this project. Two subtrees:

- `Puzzle_Perception/Segmentation/` — combined 30-class segmentation dataset
  over chess, maze, and tower-of-hanoi. Fetched from HuggingFace via
  `python download_datasets.py --dataset puzzle_perception`.
- `Puzzle_Perception/PVQA/{chess,nqueens}/` — multiple-choice perception probes for
  the CRG experiment: 900 board images, 1,200 QA rows, and cached TDDN detections.
  Shipped pre-populated (~86 MB).

See [`Puzzle_Perception/README.md`](Puzzle_Perception/README.md) for usage.

## Local-only data

`_local/` is a drop point for heavy data that is deliberately **not committed** — the
269-sample chess segmentation set and the archived CRG scratch tree. Everything in it
except its README is gitignored, so a fresh clone gets an empty directory and the few
tracks that need it fail with an explicit message. Relocate it with
`EXPERIMENTS_LOCAL_DATA_ROOT=/path/to/dir`. See [`_local/README.md`](_local/README.md)
for what belongs there and which tracks depend on it.

## License

Each public dataset retains its original license; see the upstream
page linked in the downloads table. Code in this directory is released
under the repository's root license.
