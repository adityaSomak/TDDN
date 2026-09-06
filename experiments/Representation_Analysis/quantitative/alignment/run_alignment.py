"""Wang & Isola ALIGNMENT for representation discriminability.

Two settings (alignment only — uniformity NOT computed, per scope):
  A — PATCH (SPair-71k): positives = ground-truth matched keypoints across SPair
      test pairs. Discriminability-relevant.
  B — GLOBAL (ImageNet val): positives = same-class image pairs. Labeled
      "class-alignment / semantic clustering" (a semantic property, NOT patch
      matching discriminability).

alignment  L_align = E_{(x,x+)} ||f̂(x) − f̂(x+)||²   (features L2-normalized).
For L2-normed rows this is  mean( 2 − 2·cos ).  Lower = positives closer = better.

All features extracted at 512px square_resize (CLIP at 518 = 14×37), the
Representation_Analysis protocol (DINOv3 facet=token; CleanDIFT resnets[2] t=0);
CD uses per-layer PCA→512 (CDp=1536). Trained tdn/tddn loaded from the named
flat TDN / TDDN checkpoints (configured by `checkpoint` in models.yaml).
Analysis-only; no model/other-experiment/committed file touched.

Single-file driver+worker, --smoke/--full, --setting patch|global|both.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

# Cap intra-op threads: the per-image assembly does tens of thousands of tiny
# torch.interpolate + BLAS ops; without a cap each spawns threads across all
# cores and oversubscription overhead dominates. (Set BLAS caps via env in the
# launch command; this caps torch.)
torch.set_num_threads(8)

# ── make experiments tree + this experiment importable ────────────────────────
_THIS = Path(__file__).resolve()
_ALIGN = _THIS.parent
_RA = _THIS.parents[2]
_EXPERIMENTS = _THIS.parents[3]
for _p in (_EXPERIMENTS, _RA, Path("/home/shanmukha/dinov3")):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared_utils.paths import DATASETS_ROOT, FEATURES_ROOT          # noqa: E402
from shared_utils.feature_extraction import build_extractor, build_transform, loader_kwargs_for  # noqa: E402
from metrics.extract import _extractor_view, _save_per_image          # noqa: E402
from metrics.feature_utils import (                                   # noqa: E402
    interpolate_to_32x32, build_diffusion_combined_pca, build_fused,
    _row_normalize, load_feature,
)
from metrics.quality import l2_normalize, uniformity                  # noqa: E402

_CONFIGS = _RA / "configs"
_RESULTS = _ALIGN / "results"
SPAIR_ROOT = DATASETS_ROOT / "Existing_Datasets/Keypoint_Matching/SPair-71K/SPair-71k"

# ── constants ─────────────────────────────────────────────────────────────────
TARGET = 32                 # common grid for DN/CD/SD/trained (512/16)
PCA_DIM = 512
SEED = 42
N_ERR_SEEDS = 5             # seeded subsamples for alignment mean±std
SUBSAMPLE_FRAC = 0.8
N_GPUS = 2

def input_size(tag: str) -> int:
    return 518 if tag == "clip" else 512   # CLIP patch=14 → 518=14×37

# Views to persist per tag, per setting: (layer_subdir, extractor_view_key).
PATCH_VIEWS = {
    "dinov3":  [("dinov3_patches", "patches")],
    "cd":      [("cd_layer2", "per_layer:2"), ("cd_layer5", "per_layer:5"), ("cd_layer8", "per_layer:8")],
    "sd-2.1":  [("sd-2.1_layer2", "per_layer:2"), ("sd-2.1_layer5", "per_layer:5"), ("sd-2.1_layer8", "per_layer:8")],
    "clip":    [("clip_patches", "patches")],
    "tdn":     [("tdn_patches", "patches")],
    "tddn":    [("tddn_patches", "patches")],
}
GLOBAL_VIEWS = {
    "dinov3":  [("dinov3_cls", "cls"), ("dinov3_patches", "patches")],
    "cd":      [("cd_layer2", "per_layer:2"), ("cd_layer5", "per_layer:5"), ("cd_layer8", "per_layer:8")],
    "clip":    [("clip_cls", "cls")],
    "tdn":     [("tdn_global", "global")],
    "tddn":    [("tddn_global", "global")],
}
SIZES = {"smoke": dict(spair_cats=2, spair_cap=30, in_per_class=4, in_classes=50),
         "full":  dict(spair_cats=18, spair_cap=150, in_per_class=10, in_classes=1000)}


def log(m: str) -> None:
    print(f"[align {time.strftime('%H:%M:%S')}] {m}", flush=True)


def ensure_hf_token() -> None:
    if not os.environ.get("HF_TOKEN"):
        tok = Path.home() / ".cache/huggingface/token"
        if tok.is_file():
            os.environ["HF_TOKEN"] = tok.read_text().strip()


# ══════════════════════════════════════════════════════════════════════════════
# Extractor construction (handles trained tags) + batched extraction
# ══════════════════════════════════════════════════════════════════════════════
def make_extractor(tag: str, models_cfg: dict, device):
    entry = None
    for g in ("baselines", "trained"):
        if tag in (models_cfg.get(g) or {}):
            entry = models_cfg[g][tag]
            break
    if entry is None:
        raise KeyError(f"tag {tag} not in models.yaml")
    backbone = entry["backbone"]
    isz = input_size(tag)
    ekw = dict(entry.get("extractor", {}) or {})
    lkw = dict(loader_kwargs_for(entry))
    if backbone == "fused-dinov3-cd":              # tddn
        ekw["return_patches"] = True
        lkw["common_grid_override"] = max(1, isz // 16)   # 32 @ 512
    extractor = build_extractor(backbone, device, extractor_kwargs=ekw,
                                loader_kwargs_override=lkw or None)
    transform = build_transform(backbone, isz, entry.get("transform", {}).get("strategy", "square_resize"))
    return extractor, transform


class _PathDS(torch.utils.data.Dataset):
    """Yield (transformed_tensor, stem) for a list of (stem, path)."""
    def __init__(self, items, transform):
        self.items = items; self.t = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        stem, path = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.t(img), stem


def _batch_for(tag: str) -> int:
    return {"dinov3": 24, "clip": 24, "cd": 10, "sd-2.1": 10, "tdn": 16, "tddn": 8}.get(tag, 12)


def extract_tag(tag, items, views, features_root, split, device, models_cfg):
    """Batched extraction for one tag over (stem,path) items → per-image .npy."""
    todo = [(s, p) for (s, p) in items
            if not all((features_root / lay / split / f"{s}.npy").is_file() for lay, _ in views)]
    if not todo:
        log(f"  [{tag}] all cached"); return
    extractor, transform = make_extractor(tag, models_cfg, device)
    bs = _batch_for(tag)
    dl = torch.utils.data.DataLoader(_PathDS(todo, transform), batch_size=bs,
                                     num_workers=6, collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b]))
    n = 0
    for imgs, stems in dl:
        with torch.no_grad():
            out = extractor.extract(imgs.to(device))
        for lay, key in views:
            flat = _extractor_view(out, key)          # (B, N, C)
            arr = flat.cpu().numpy()
            for b, stem in enumerate(stems):
                _save_per_image(arr[b], features_root / lay / split, stem)
        n += len(stems)
        if n % 480 == 0:
            log(f"  [{tag}] {n}/{len(todo)}")
    del extractor
    torch.cuda.empty_cache()
    log(f"  [{tag}] done ({len(todo)})")


def run_worker(shard, num_shards, items_file, split, tags):
    ensure_hf_token()
    items = [tuple(l.split("\t")) for l in Path(items_file).read_text().splitlines() if l]
    items = items[shard::num_shards]
    views = PATCH_VIEWS if split == "spair" else GLOBAL_VIEWS
    models_cfg = yaml.safe_load((_CONFIGS / "models.yaml").read_text())
    log(f"worker {shard}/{num_shards}: {len(items)} imgs on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for tag in tags:
        extract_tag(tag, items, views[tag], FEATURES_ROOT, split, "cuda", models_cfg)


def run_extraction(items, split, tags):
    """items: list of (stem, path). Spawn 1 worker per GPU (batched saturates a GPU)."""
    _RESULTS.mkdir(parents=True, exist_ok=True)
    f = _RESULTS / f"_items_{split}.tsv"
    f.write_text("\n".join(f"{s}\t{p}" for s, p in items))
    procs = []
    for w in range(N_GPUS):
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(w)
        env["PYTHONPATH"] = "/home/shanmukha/dinov3:" + env.get("PYTHONPATH", "")
        procs.append(subprocess.Popen(
            [sys.executable, str(_THIS), "--worker", "--shard", str(w),
             "--num-shards", str(N_GPUS), "--items-file", str(f),
             "--split", split, "--tags", ",".join(tags)], env=env))
    rcs = [p.wait() for p in procs]
    if any(rcs):
        raise RuntimeError(f"extraction workers failed: {rcs}")
    log(f"extraction done ({split})")


# ══════════════════════════════════════════════════════════════════════════════
# Alignment
# ══════════════════════════════════════════════════════════════════════════════
def alignment(A, B):
    """E[||f̂(a)-f̂(b)||²] = mean(2 - 2·cos) over L2-normed rows."""
    An = l2_normalize(np.asarray(A, np.float64)); Bn = l2_normalize(np.asarray(B, np.float64))
    return float(np.mean(2.0 - 2.0 * np.sum(An * Bn, axis=1)))


def alignment_meanstd(A, B):
    A = np.asarray(A, np.float32); B = np.asarray(B, np.float32)
    n = len(A); k = max(2, int(SUBSAMPLE_FRAC * n))
    vals = []
    for s in range(N_ERR_SEEDS):
        idx = np.random.default_rng(SEED + s).choice(n, size=k, replace=False)
        vals.append(alignment(A[idx], B[idx]))
    return round(float(np.mean(vals)), 6), round(float(np.std(vals)), 6), n


def uniformity_meanstd(X, n_sub=10000):
    """Wang & Isola uniformity (t=2), reused from metrics.quality. Mean±std over
    N_ERR_SEEDS subsamples (subsample < N gives error bars). Lower = more uniform."""
    X = np.asarray(X, np.float32)
    n = len(X)
    sub = min(n_sub, max(2, int(SUBSAMPLE_FRAC * n)))   # ensure subsampling for error bars
    vals = [uniformity(X, n_subsample=sub, seed=SEED + s) for s in range(N_ERR_SEEDS)]
    return round(float(np.mean(vals)), 6), round(float(np.std(vals)), 6), n


# ══════════════════════════════════════════════════════════════════════════════
# CD/SD per-layer PCA basis (fit on patch tokens of this dataset)
# ══════════════════════════════════════════════════════════════════════════════
def fit_layer_pca(stems, split, prefix, n_fit_imgs=400, n_sub=64):
    """Fit 3 per-layer PCA(512) on a seeded sample of patch tokens.
    prefix: 'cd' or 'sd-2.1'. Returns reducers dict."""
    layers = [f"{prefix}_layer2", f"{prefix}_layer5", f"{prefix}_layer8"]
    rng = np.random.default_rng(SEED)
    pick = stems if len(stems) <= n_fit_imgs else [stems[i] for i in sorted(rng.choice(len(stems), n_fit_imgs, replace=False))]
    mats = []
    for lay in layers:
        rows = []
        for st in pick:
            g = interpolate_to_32x32(load_feature(str(FEATURES_ROOT / lay / split / f"{st}.npy")), TARGET)
            ii = rng.choice(TARGET * TARGET, size=min(n_sub, TARGET * TARGET), replace=False)
            rows.append(g[ii])
        mats.append(np.concatenate(rows, 0))
    _, reducers = build_diffusion_combined_pca(*mats, PCA_DIM, reducers=None,
                                               layer_keys=(f"{prefix}2", f"{prefix}5", f"{prefix}8"))
    log(f"  PCA[{prefix}] fit on {len(pick)}×{n_sub} patches → {[reducers[k].actual_n_components for k in reducers]}")
    return reducers


def cd_grid(stem, split, prefix, reducers):
    """Per-image CDp grid (1024,1536): interp each layer→32, PCA-transform→L2→concat."""
    keys = (f"{prefix}2", f"{prefix}5", f"{prefix}8")
    parts = []
    for lay, k in zip((f"{prefix}_layer2", f"{prefix}_layer5", f"{prefix}_layer8"), keys):
        g = interpolate_to_32x32(load_feature(str(FEATURES_ROOT / lay / split / f"{stem}.npy")), TARGET)
        parts.append(_row_normalize(reducers[k].transform(g)))
    return np.concatenate(parts, 1)            # (1024, 1536)


def grid_load(stem, split, layer, G):
    """Load a cached patch grid as (G*G, C); for DN/trained G=32 already, CLIP G=37."""
    a = load_feature(str(FEATURES_ROOT / layer / split / f"{stem}.npy"))
    hw = a.shape[0]; g = int(round(hw ** 0.5))
    return a if g == G else interpolate_to_32x32(a, G)


def kp_idx(x, y, W, H, G):
    c = min(int(x / W * G), G - 1); r = min(int(y / H * G), G - 1)
    return max(r, 0) * G + max(c, 0)


# ══════════════════════════════════════════════════════════════════════════════
# Setting A — PATCH (SPair)
# ══════════════════════════════════════════════════════════════════════════════
def load_spair_pairs(cats, cap):
    """Replicate the PCK pair loader: mirror-filtered test pairs, capped/cat.
    Returns (pairs, stem→path). pair = (cat, s_stem,s_kps,s_wh, t_stem,t_kps,t_wh)."""
    layout = [l.strip() for l in (SPAIR_ROOT / "Layout/large/test.txt").read_text().splitlines() if l.strip()]
    by_cat = {}
    for rel in layout:
        j = json.loads((SPAIR_ROOT / "PairAnnotation/test" / f"{rel}.json").read_text())
        if j.get("mirror", 0) != 0:
            continue
        by_cat.setdefault(j["category"], []).append(j)
    chosen_cats = sorted(by_cat)[:cats]
    rng = np.random.default_rng(SEED)
    pairs, stem2path = [], {}
    def stem(cat, imname): return f"{cat}__{Path(imname).stem}"
    for cat in chosen_cats:
        js = by_cat[cat]
        if len(js) > cap:
            js = [js[i] for i in sorted(rng.choice(len(js), cap, replace=False))]
        for j in js:
            ss, ts = stem(cat, j["src_imname"]), stem(cat, j["trg_imname"])
            stem2path[ss] = SPAIR_ROOT / "JPEGImages" / cat / j["src_imname"]
            stem2path[ts] = SPAIR_ROOT / "JPEGImages" / cat / j["trg_imname"]
            pairs.append((cat, ss, np.asarray(j["src_kps"], float), tuple(j["src_imsize"][:2]),
                          ts, np.asarray(j["trg_kps"], float), tuple(j["trg_imsize"][:2])))
    return pairs, stem2path


# rep → (kind, grid G). kind: dn|cd|sd|ddn|raw32|clip|trained32
PATCH_REPS = [("DNp", "dn", 32, 1280), ("CDp", "cd", 32, 1536), ("DDNp", "ddn", 32, 2816),
              ("SDp", "sd", 32, 1536), ("CLIPp", "clip", 37, 1024),
              ("TDNp", "raw", 32, 1280), ("TDDNp", "raw", 32, 1280)]
RAW_LAYER = {"DNp": "dinov3_patches", "CLIPp": "clip_patches", "TDNp": "tdn_patches", "TDDNp": "tddn_patches"}


def patch_pipeline(sz, records, skip_extract):
    pairs, stem2path = load_spair_pairs(sz["spair_cats"], sz["spair_cap"])
    stems = sorted(stem2path)
    log(f"SPair: {len(pairs)} pairs, {len(stems)} unique imgs, {sz['spair_cats']} cats")
    items = [(s, str(stem2path[s])) for s in stems]
    tags = ["dinov3", "cd", "sd-2.1", "clip", "tdn", "tddn"]
    if not skip_extract:
        run_extraction(items, "spair", tags)
    cd_red = fit_layer_pca(stems, "spair", "cd")
    sd_red = fit_layer_pca(stems, "spair", "sd-2.1")

    for name, kind, G, d in PATCH_REPS:
        cache = {}
        def grid(stem):
            if stem not in cache:
                if kind == "cd":
                    cache[stem] = cd_grid(stem, "spair", "cd", cd_red)
                elif kind == "sd":
                    cache[stem] = cd_grid(stem, "spair", "sd-2.1", sd_red)
                elif kind == "ddn":
                    dn = grid_load(stem, "spair", "dinov3_patches", 32)
                    cache[stem] = build_fused(dn, cd_grid(stem, "spair", "cd", cd_red))
                elif kind in ("dn", "raw", "clip"):
                    cache[stem] = grid_load(stem, "spair", RAW_LAYER[name], G)
            return cache[stem]
        A, B = [], []
        for cat, ss, skp, swh, ts, tkp, twh in pairs:
            gs, gt = grid(ss), grid(ts)
            for i in range(len(skp)):
                A.append(gs[kp_idx(skp[i, 0], skp[i, 1], swh[0], swh[1], G)])
                B.append(gt[kp_idx(tkp[i, 0], tkp[i, 1], twh[0], twh[1], G)])
        m, s, n = alignment_meanstd(A, B)
        # uniformity over the population of keypoint features (src ∪ trg) — same
        # features the alignment operates on.
        pop = np.vstack([np.asarray(A, np.float32), np.asarray(B, np.float32)])
        um, us, npts = uniformity_meanstd(pop)
        records.append(dict(setting="patch", representation=name, alignment_mean=m, alignment_std=s,
                            uniformity_mean=um, uniformity_std=us, n_pairs=n, n_points=npts,
                            d=int(np.asarray(A[0]).shape[0]),
                            resolution=input_size("clip" if name == "CLIPp" else "dinov3"),
                            seed=SEED, positives_definition="SPair matched keypoints",
                            uniformity_population="SPair keypoint features (src∪trg)", note=""))
        log(f"  {name}: align={m:.4f}±{s:.4f} unif={um:.4f}±{us:.4f} n_pairs={n} n_pts={npts} d={records[-1]['d']}")
        cache.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Setting B — GLOBAL (ImageNet) — class-alignment / semantic clustering
# ══════════════════════════════════════════════════════════════════════════════
NOTE_GLOBAL = ("class-alignment / semantic clustering — measures same-class clustering, "
               "a semantic property, NOT patch-matching discriminability.")


def imagenet_items(sz, models_cfg):
    """Materialize the balanced val subset to JPEGs on disk (so the shared
    extraction path can consume file paths), return (items, labels{stem}).

    Downloads ONLY the 14 validation parquet shards (~6 GB) via hf_hub_download
    — NOT load_dataset(split=...), which pulls all 278 train shards (~130 GB)
    first and stalls."""
    from huggingface_hub import HfApi, hf_hub_download
    from datasets import load_dataset
    tok = os.environ.get("HF_TOKEN")
    files = [f for f in HfApi().list_repo_files("ILSVRC/imagenet-1k", repo_type="dataset", token=tok)
             if "validation-" in f and f.endswith(".parquet")]
    log(f"downloading {len(files)} val parquet shards (~6 GB)…")
    paths = [hf_hub_download("ILSVRC/imagenet-1k", f, repo_type="dataset", token=tok) for f in files]
    ds = load_dataset("parquet", data_files={"validation": paths}, split="validation")
    labels_all = ds["label"]
    by_cls = {}
    for i, l in enumerate(labels_all):
        by_cls.setdefault(int(l), []).append(i)
    rng = np.random.default_rng(SEED)
    img_dir = _RESULTS / "imagenet_jpg"; img_dir.mkdir(parents=True, exist_ok=True)
    items, labels = [], {}
    for cls in sorted(by_cls)[: sz["in_classes"]]:
        pool = by_cls[cls]; rng.shuffle(pool)
        for idx in sorted(pool[: sz["in_per_class"]]):
            stem = f"in_{idx:06d}"
            p = img_dir / f"{stem}.jpg"
            if not p.is_file():
                im = ds[idx]["image"]
                (im.convert("RGB") if im.mode != "RGB" else im).save(p, "JPEG", quality=95)
            items.append((stem, str(p))); labels[stem] = cls
    log(f"ImageNet: {len(items)} imgs, {sz['in_classes']} classes × {sz['in_per_class']}")
    return items, labels


def patch_mean_grid_dn(stem):  # DN patch mean (1280)
    return grid_load(stem, "imagenet", "dinov3_patches", 32).mean(0)


def global_pipeline(sz, records, skip_extract):
    models_cfg = yaml.safe_load((_CONFIGS / "models.yaml").read_text())
    items, labels = imagenet_items(sz, models_cfg)
    stems = [s for s, _ in items]
    tags = ["dinov3", "cd", "clip", "tdn", "tddn"]
    if not skip_extract:
        run_extraction(items, "imagenet", tags)
    cd_red = fit_layer_pca(stems, "imagenet", "cd")

    def ld(layer, stem):
        return load_feature(str(FEATURES_ROOT / layer / "imagenet" / f"{stem}.npy"))

    GLOBAL_REPS = ["DNg", "CDpbar", "DDNg", "CLIPg", "TDNg", "TDDNg"]
    feats = {name: {} for name in GLOBAL_REPS}
    for i, st in enumerate(stems):
        cdp = cd_grid(st, "imagenet", "cd", cd_red)              # (1024,1536) — computed ONCE
        cls = ld("dinov3_cls", st)[0]                            # (1280,)
        dn = grid_load(st, "imagenet", "dinov3_patches", 32)     # (1024,1280)
        ddnp = build_fused(dn, cdp)                              # (1024,2816)
        feats["DNg"][st] = cls
        feats["CDpbar"][st] = cdp.mean(0)                        # (1536,)
        feats["DDNg"][st] = np.concatenate([_row_normalize(cls[None])[0], ddnp.mean(0)])  # 4096
        feats["CLIPg"][st] = ld("clip_cls", st)[0]
        feats["TDNg"][st] = ld("tdn_global", st)[0]
        feats["TDDNg"][st] = ld("tddn_global", st)[0]
        if (i + 1) % 2000 == 0:
            log(f"  assembled {i + 1}/{len(stems)} global vectors")

    # same-class pairs
    by_cls = {}
    for st in stems:
        by_cls.setdefault(labels[st], []).append(st)
    for name in GLOBAL_REPS:
        A, B = [], []
        for cls, sts in by_cls.items():
            for a in range(len(sts)):
                for b in range(a + 1, len(sts)):
                    A.append(feats[name][sts[a]]); B.append(feats[name][sts[b]])
        m, s, n = alignment_meanstd(A, B)
        # uniformity over the global-vector population (all images, all classes).
        pop = np.stack([feats[name][st] for st in stems])
        um, us, npts = uniformity_meanstd(pop)
        records.append(dict(setting="global", representation=name, alignment_mean=m, alignment_std=s,
                            uniformity_mean=um, uniformity_std=us, n_pairs=n, n_points=npts,
                            d=int(np.asarray(A[0]).shape[0]), resolution=512, seed=SEED,
                            positives_definition="ImageNet same-class",
                            uniformity_population="ImageNet global-vector pool", note=NOTE_GLOBAL))
        log(f"  {name}: class-align={m:.4f}±{s:.4f} unif={um:.4f}±{us:.4f} n_pairs={n} n_pts={npts} d={records[-1]['d']}")


# ══════════════════════════════════════════════════════════════════════════════
def write_outputs(records, mode):
    import pandas as pd
    _RESULTS.mkdir(parents=True, exist_ok=True)
    suf = "_smoke" if mode == "smoke" else ""
    jp = _RESULTS / f"alignment{suf}.json"
    # Merge: preserve rows for settings NOT computed this run (e.g. keep cached
    # patch rows when running --setting global only).
    done = {r["setting"] for r in records}
    if jp.is_file():
        try:
            prev = [r for r in json.loads(jp.read_text()) if r["setting"] not in done]
            records = prev + records
        except Exception:
            pass
    order = {("patch", x): i for i, x in enumerate(["DNp", "CDp", "DDNp", "SDp", "CLIPp", "TDNp", "TDDNp"])}
    order.update({("global", x): 100 + i for i, x in enumerate(["DNg", "CDpbar", "DDNg", "CLIPg", "TDNg", "TDDNg"])})
    records.sort(key=lambda r: order.get((r["setting"], r["representation"]), 999))
    jp.write_text(json.dumps(records, indent=2))
    pd.DataFrame(records).to_csv(_RESULTS / f"alignment{suf}.csv", index=False)
    print("\n" + "=" * 86)
    print(f"  ALIGNMENT + UNIFORMITY (mode={mode})  — alignment: lower=better; uniformity: lower=more spread")
    print("=" * 86)
    print(f"  {'setting':7} {'rep':8} {'alignment ↓':>18}  {'uniformity ↓':>18}  {'n_pairs':>8} {'d':>5}")
    print("-" * 86)
    for r in records:
        print(f"  {r['setting']:7} {r['representation']:8} "
              f"{r['alignment_mean']:.4f} ± {r['alignment_std']:.4f}   "
              f"{r['uniformity_mean']:.4f} ± {r['uniformity_std']:.4f}   "
              f"{r['n_pairs']:>8} {r['d']:>5}")
    print("=" * 86 + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=N_GPUS)
    ap.add_argument("--items-file", default=""); ap.add_argument("--split", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true")
    ap.add_argument("--setting", choices=["patch", "global", "both"], default="both")
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()

    if a.worker:
        run_worker(a.shard, a.num_shards, a.items_file, a.split, a.tags.split(","))
        return
    if a.smoke == a.full:
        raise SystemExit("specify exactly one of --smoke / --full")
    ensure_hf_token()
    mode = "smoke" if a.smoke else "full"
    sz = SIZES[mode]
    log(f"mode={mode} setting={a.setting} sizes={sz}")
    records = []
    if a.setting in ("patch", "both"):
        patch_pipeline(sz, records, a.skip_extract)
    if a.setting in ("global", "both"):
        global_pipeline(sz, records, a.skip_extract)
    write_outputs(records, mode)


if __name__ == "__main__":
    main()
