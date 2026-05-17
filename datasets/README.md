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
│   │   └── ADE20K/
│   ├── Vision_Language_Alignment/
│   │   ├── LAION-5B/
│   │   └── MS-COCO-2014/
│   └── PVQA/
│       ├── AlgoPuzzleVQA/
│       └── AlgoPuzzleVQA_star/
└── Puzzle_Perception/
    ├── Segmentation/                 # combined 30-class seg dataset (HF download)
    └── PVQA/test/chess/              # 269-sample chess VQA + mask overlays
```

## Requirements

```bash
pip install torch torchvision datasets
```

Some datasets need additional tools (`curl`, `tar`, `unzip`) that ship with
most Unix systems.

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
| Caltech-101        | `Existing_Datasets/Classification/Caltech-101/`      | torchvision                                                                                  |
| Food-101           | `Existing_Datasets/Classification/Food-101/`         | torchvision                                                                                  |
| GTSRB              | `Existing_Datasets/Classification/GTSRB/`            | torchvision                                                                                  |
| ImageNet-1K        | `Existing_Datasets/Classification/ImageNet-1K/`      | [`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k) (gated)           |
| SPair-71K          | `Existing_Datasets/Keypoint_Matching/SPair-71K/`     | [postech.ac.kr](https://cvlab.postech.ac.kr/research/SPair-71k/)                             |
| Flickr30K          | `Existing_Datasets/Retrieval/Flickr30K/`             | [`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k)                     |
| ADE20K             | `Existing_Datasets/Segmentation/ADE20K/`             | [data.csail.mit.edu](http://data.csail.mit.edu/places/ADEchallenge/)                         |
| MS-COCO-2014       | `Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014/` | [images.cocodataset.org](http://images.cocodataset.org/)                              |
| Recaptioned LAION  | `Existing_Datasets/Vision_Language_Alignment/LAION-5B/data/` | [`PuzzleBench/Recaptioned_LAION`](https://huggingface.co/datasets/PuzzleBench/Recaptioned_LAION) (508k samples, ~72 GB, 30 webdataset shards) |
| Puzzle-Perception  | `Puzzle_Perception/Segmentation/data/`               | [`PuzzleBench/Puzzle_Perception`](https://huggingface.co/datasets/PuzzleBench/Puzzle_Perception) (combined 30-class segmentation) |

### Authentication

Gated datasets (ImageNet-1K and the `PuzzleBench/*` HuggingFace repos while
they remain private) require a HuggingFace access token:

```bash
export HF_TOKEN=hf_...
```

Request access at the dataset page before downloading.

## PVQA: AlgoPuzzleVQA

This repository ships the PVQA splits used in our evaluations.

### `AlgoPuzzleVQA/`

Six algorithmic puzzle tasks — `checker_move`, `maze_solve`, `nqueens`,
`tower_of_hanoi`, `water_jugs`, `wood_slide` — each shaped as:

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

The upstream benchmark is described in
[AlgoPuzzleVQA](https://github.com/declare-lab/LLM-PuzzleTest/tree/master/AlgoPuzzleVQA).

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
- `Puzzle_Perception/PVQA/test/chess/` — 269 chess samples used in the
  visual-question-answering evaluation, paired with oracle and TDDN mask
  overlays. Shipped pre-populated.

See [`Puzzle_Perception/README.md`](Puzzle_Perception/README.md) for usage.

## License

Each public dataset retains its original license; see the upstream page
linked in the table. PVQA assets follow the license of the AlgoPuzzleVQA
project. Code in this directory is released under the repository's root
license.
