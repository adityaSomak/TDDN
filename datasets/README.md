
### Structure of `datasets` directory

```
datasets/
├── Existing_Datasets/
│   ├── Classification/         # CIFAR-100, Caltech-101, Food-101, GTSRB, ImageNet-1K
│   ├── Keypoint_Matching/      # SPair-71K
│   ├── Retrieval/              # Flickr30K
│   ├── Segmentation/           # ADE20K
│   ├── Vision_Language_Alignment/ # LAION-5B, MS-COCO-2014
│   └── PVQA/
│       ├── AlgoPuzzleVQA/
│       └── AlgoPuzzleVQA_star/
│           ├── maze/
│           │   ├── data/       # csv, jsonl, and images
│           │   └── scripts/    # Generation and answer scripts
│           └── nqueens/
│               ├── data/       # Fixed .csv extension
│               └── scripts/
│
├── Puzzle_Perception/          # Custom/Novel Data
│   ├── PVQA/
│   │   └── test/
│   │       └── chess/
│   └── Segmentation/
│       ├── train/
│       │   ├── chess/
│       │   ├── maze/
│       │   └── tower_of_hanoi/
│       ├── val/
│       └── test/
└── README.md
```
