# Qualitative — activation maps

Per-image PCA(3) → RGB visualization of every registered backbone.

```
qualitative/
├── samples/                                    # input images
├── baselines/activation-maps/
│   ├── sd-2.1/
│   ├── cd/
│   ├── dinov3/
│   ├── dinov2-vitb/
│   ├── dinov2-vitg/
│   └── clip/
├── tdn/activation-maps/dinov3+roberta/
├── tddn/activation-maps/dinov3+cd+roberta/
└── ddn/activation-maps/dinov3+cd/
```

Each leaf directory holds `<image_stem>.png` and `<image_stem>.pdf` at
300 dpi. Render defaults (input=1024, target=512, mode=patches) live in
`configs/activation_maps.yaml`; the per-model extractor and transform
kwargs live in `configs/models.yaml`.

## Add a new model

1. Add an entry to `configs/models.yaml` under `baselines`, `trained`, or
   `fusion`.
2. Add the leaf directory under `qualitative/<group>/activation-maps/`.
3. Re-run `python run.py activation-maps --image all --model <tag>`.

## Add a new sample image

Drop the image under `samples/` and append its filename to
`configs/activation_maps.yaml:images`.
