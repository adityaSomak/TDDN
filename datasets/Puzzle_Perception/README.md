# Puzzle Perception

Custom data for the puzzle-perception research track. Two subtrees:

| Subtree | Purpose | Storage |
|---|---|---|
| [`PVQA/`](PVQA/) | Multiple-choice perception probes over chess and N-Queens boards, for the CRG experiment: board images, question specs, answer CSVs and cached TDDN detections. | **shipped in the repo** (~86 MB, 900 boards + 1,200 QA rows) — no download step |
| [`Segmentation/`](Segmentation/) | Combined per-pixel segmentation dataset over three puzzle domains (chess, maze, tower-of-hanoi) under a single 30-class label space. | **schema only** in the repo (`classes.yaml` + `dataset.py`); the image/mask data is fetched from HuggingFace via `download_datasets.py` |

The older 269-sample chess seg-eval set that used to live at `PVQA/test/chess/` is no
longer committed; it moved to [`../_local/chess_seg269/`](../_local/README.md), which is
gitignored and must be supplied locally. It is still needed by the
`Puzzle_Understanding` chess seg track and by CRG chess board regeneration — but not
by CRG evaluation.

## Layout

```
Puzzle_Perception/
├── README.md                              # this file
│
├── PVQA/                                  # shipped in the repo — see PVQA/README.md
│   ├── chess/
│   │   ├── questions.yaml                 # 8 probes (eval spec + generation spec)
│   │   ├── answers.csv                    # 800 rows
│   │   ├── tddn_detections.json           # cached TDDN 8x8 class maps
│   │   └── images/<image_id>.png          # 800 generated boards, 512x512
│   └── nqueens/
│       ├── questions.yaml                 # 4 probes
│       ├── answers.csv                    # 400 rows (100 boards x 4 questions)
│       ├── tddn_detections.json           # cached TDDN queen boxes
│       └── images/<image_id>.jpg          # 100 source boards
│
└── Segmentation/
    ├── classes.yaml                       # shipped — 30 unified classes
    ├── dataset.py                         # shipped — PuzzleSegmentationDataset
    └── data/                              # downloaded — populated by ../download_datasets.py
        ├── manifest.csv
        └── {train,val,test}/{images,masks}/<task>_<id>.png
```

The repo-wide [`requirements.txt`](../../requirements.txt) supplies the
deps you need to load either subtree.

## PVQA — CRG perception probes

Board images, question specs, answer CSVs and cached TDDN detections for the two CRG
tasks, all committed. See the [dataset card](PVQA/README.md) for the CSV/YAML schemas,
the naming convention and the validation command.

| Task | Boards | Board size | Questions | Rows |
|---|---|---|---|---|
| `chess` | 800 generated | 8×8, 512 px | 8 (`q1`–`q8`), 100 boards each | 800 |
| `nqueens` | 100 from AlgoPuzzleVQA* | 8×8 – 11×11 | 4 (`q1`–`q4`), all boards each | 400 |

CRG negatives (the region-blacked images) are **not** shipped — they are rebuilt in
memory at eval time from the board plus either the GT `ablate_cells` or the cached
detections. Only the detections are stored, which is what lets the predicted-region arm
be reproduced without a GPU. The eval itself lives under
[`experiments/CRG/`](../../experiments/CRG/README.md).

## Segmentation — 30-class unified label space

Only the schema files (`classes.yaml`, `dataset.py`) ship in the repo;
the image/mask splits are fetched from HuggingFace:

```bash
# from the datasets/ root
python download_datasets.py --dataset puzzle_perception
```

Then load it from Python:

```python
from datasets.Puzzle_Perception.Segmentation.dataset import (
    PuzzleSegmentationDataset,
)

ds = PuzzleSegmentationDataset(root="Segmentation/data", split="train")
image, mask, meta = ds[0]
# image: float tensor [3, H, W], values in [0, 1]
# mask:  long tensor [H, W], values in [0, 29]
# meta:  {"unified_id": "chess_000000", "source_task": "chess", "source_id": "000000"}

print(f"{len(ds)} samples, {ds.classes.num_classes} classes")
print(ds.classes.names()[:5])
# ['wall', 'path', 'start', 'dest_a', 'dest_b']
```

The `Dataset` exposes class metadata via a `Classes` helper:
`ds.classes.ce_weights()`, `ds.classes.miou_weights()`,
`ds.classes.ids_for_task("chess")`, etc.

### Unified label space

Masks are 8-bit PNGs with values `0..29`. The mapping is defined in
[`Segmentation/classes.yaml`](Segmentation/classes.yaml):

| Source task | Class ids | Count |
|---|---|---|
| maze   |  0..7    |  8 |
| chess  |  8..22   | 15 |
| hanoi  | 23..29   |  7 |

Every sample's filename encodes its source task
(`chess_000000.png`, `maze_00012.png`, `hanoi_00042.png`); the same
information is also recorded per-row in `data/manifest.csv` and (after
HuggingFace upload) as the `source_task` feature column.

### Default subset sizes

The published dataset is a deterministically-sampled subset of the raw
sources rather than the full set, matching the segmentation-probe
training recipe. Per source task:

| Split | Samples per task | Total |
|---|---:|---:|
| `train` | 2000 | 6000 |
| `val`   |  500 | 1500 |
| `test`  |  500 | 1500 |

Sampling uses a fixed seed (`42`), so the subset is reproducible.

## Provenance

- **Segmentation source datasets** — the three raw datasets
  (`chess_dataset`, `natural_maze_dataset`, `hanoi_dataset`) were
  generated by the authors and combined into a single 30-class corpus
  for this release.
- **PVQA chess overlays** — `oracle_mask/` is derived deterministically
  from the ground-truth segmentation masks; `tddn_mask/` is produced by
  the TDDN tip-adapter pipeline.
