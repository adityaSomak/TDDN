# MS-COCO 2014 — drop-point

The Representation_Analysis pipeline iterates a fixed 2000-image subset of
COCO val2014. Image data is **not** committed (6.3 GB); place it here yourself.

## Expected layout

```
MS-COCO-2014/
└── val2014/
    ├── COCO_val2014_000000000042.jpg
    ├── COCO_val2014_000000000073.jpg
    └── ... (40,504 jpg files)
```

The 2000-image subset is fixed by stems in
`experiments/Representation_Analysis/configs/coco_sample_ids.csv`. Every
clone gets identical sampling for free; nothing here is random.

## Getting the data

```bash
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d <this dir>
# → val2014/COCO_val2014_*.jpg
```

Or symlink an existing local copy:

```bash
ln -s /path/to/your/coco/val2014 val2014
```

## Resolution

The path resolves as
`shared_utils.paths.DATASETS_ROOT / "Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014/val2014"`,
which by default is `<repo>/datasets/.../MS-COCO-2014/val2014/`. Override
the root with `EXPERIMENTS_DATASETS_ROOT` if your dataset tree lives outside
the repo.
