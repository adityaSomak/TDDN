# Local-only data

Heavy data that is deliberately **not committed**. Everything in this directory except
this README and `.gitkeep` is gitignored, so a fresh clone gets an empty tree and the
capabilities listed below are unavailable until you re-supply the contents.

Relocate the whole tree with `EXPERIMENTS_LOCAL_DATA_ROOT=/path/to/dir` (see
[`../../experiments/shared_utils/paths.py`](../../experiments/shared_utils/paths.py)).

## Expected contents

```
_local/
├── chess_seg269/                   # the 269-sample chess segmentation set (~272 MB)
│   ├── data/
│   │   ├── images/<pid>.png        # 269 real rendered boards, 512x512
│   │   ├── masks/<pid>.png         # 269 per-pixel GT masks (8-bit, chess-local ids 0..14)
│   │   └── text_repr.json          # GT board grids + piece counts, 269 entries
│   └── seg_data/
│       ├── oracle_mask/<pid>_overlay.jpg
│       └── tddn_mask/<pid>_overlay.jpg
└── CRG_archive/                    # pre-restructure CRG scratch (~251 MB), reference only
```

## What needs `chess_seg269/`

| Capability | Reads |
|---|---|
| `Puzzle_Understanding` chess seg-eval (`--tasks chess_count chess_grid`) | `data/images/`, `data/text_repr.json`, `seg_data/<mode>/` |
| CRG chess board **regeneration** (`run_generate.py`) | `data/images/`, `data/masks/` — piece sprites and board colour themes are extracted from the real boards |
| CRG `--redetect` for chess | `data/images/`, `data/masks/` — the TDDN Tip-Adapter support cache is built from real GT masks |

Each of those entry points fails with an explicit message naming this directory rather
than silently producing empty output.

**CRG evaluation does not need any of this.** Its dataset — board images, questions,
answers and cached TDDN detections — is committed under
[`../Puzzle_Perception/PVQA/`](../Puzzle_Perception/PVQA/), so the published numbers are
reproducible from a clone alone.

## `CRG_archive/`

The superseded scratch tree from before the CRG restructure: earlier generator
iterations, ~2,600 stale board renders, run logs, and the pre-migration result JSONs.
Retained only as provenance. The 8 paper models' results and both TDDN detection caches
have already been migrated out of it into the committed tree, so nothing here is on any
live code path. Its internal READMEs are stale — file lists in them name scripts that no
longer exist.

## Obtaining the data

Not currently downloadable: `chess_seg269/` is project-generated and has no upstream
host. If you need it and do not have it, it must be copied from a machine that does.
Publishing it to the `PuzzleBench` HuggingFace org and adding a
`download_datasets.py --dataset chess_seg269` entry (the `_hf_snapshot` + tar-extract
pattern already used for `puzzle_perception`) is the natural fix and would remove this
directory's only real drawback.
