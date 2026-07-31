"""Cross-prediction (source→source) analysis: DINOv3 (DN) ↔ CleanDIFT (CD).

Measures intrinsic, target-free non-redundancy between the two encoders that feed
the DDN fusion by asking how much of one encoder's representation is *linearly
recoverable* from the other's (RidgeCV, both directions, 5-fold CV). Low R² ⇒
non-redundant feature spaces. The regression target is the *other source block*,
never a task label and never the fused DDN vector.

Single-file driver. Modes:
  (default, driver)  download COCO val2014 (val only) → build seeded sample list
                     → spawn sharded extract workers (2 per GPU) → build the four
                     source matrices → RidgeCV cross-prediction → write outputs.
  --worker           extract dinov3+cd features for one stem shard (spawned by driver).
Size switch:
  --smoke            tiny end-to-end test (24 global imgs; 8×20 patch rows).
  --full             paper run (5000 global imgs; 1000×100 patch rows).

Analysis-side only. Reuses metrics.{extract,feature_utils}; touches no shared code,
no committed CSV, no config file.
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
import pandas as pd
import yaml

# ── Make the experiments tree + this experiment importable ────────────────────
_THIS = Path(__file__).resolve()
_CROSS = _THIS.parent                 # quantitative/cross_prediction
_RA = _THIS.parents[2]                # Representation_Analysis
_EXPERIMENTS = _THIS.parents[3]       # experiments
for _p in (_EXPERIMENTS, _RA):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared_utils.paths import DATASETS_ROOT, FEATURES_ROOT  # noqa: E402
from metrics.extract import extract_features                  # noqa: E402
from metrics.feature_utils import (                           # noqa: E402
    build_global_matrix,
    build_patch_matrix_with_indices,
    build_diffusion_combined_pca,
    make_patch_indices,
    load_feature,
    interpolate_to_32x32,
    _row_normalize,
)

_CONFIGS = _RA / "configs"
_RESULTS = _CROSS / "results"
COCO_DIR = DATASETS_ROOT / "Existing_Datasets/Vision_Language_Alignment/MS-COCO-2014/val2014"

# ── Fixed analysis hyperparameters ────────────────────────────────────────────
INPUT_SIZE = 512                       # extraction resolution (DINOv3 grid = 32×32)
PCA_DIM = 512                          # per-CD-layer TruncatedSVD target
TARGET = 32                            # common spatial grid
SEED = 42
# Widened from the task's logspace(-1,3,9): every fold selected the old ceiling
# (1000), so the optimum was boundary-limited. This grid lets RidgeCV settle on
# an interior alpha; if it still hits the top, widen further.
ALPHAS = np.logspace(-1, 7, 17)
TAGS = ["dinov3", "cd"]
CD_LAYER_KEYS = ("cd2", "cd5", "cd8")
NUM_SHARDS = 4                         # 4 workers, 2 per GPU
N_GPUS = 2

SIZES = {
    # fit_patch_sub = patches/image sampled to FIT the per-layer CD PCA basis.
    "smoke": dict(n_global=24, n_patch_images=8, n_patch_sub=20, fit_patch_sub=20),
    "full":  dict(n_global=5000, n_patch_images=1000, n_patch_sub=100, fit_patch_sub=50),
}
CD_LAYERS = (("cd_layer2", "cd2"), ("cd_layer5", "cd5"), ("cd_layer8", "cd8"))


# ══════════════════════════════════════════════════════════════════════════════
# Setup helpers
# ══════════════════════════════════════════════════════════════════════════════
def log(msg: str) -> None:
    print(f"[xpred {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_hf_token() -> None:
    """DINOv3 is gated; the loader reads HF_TOKEN from the env. Promote the
    cached CLI token to the env var if not already set."""
    if os.environ.get("HF_TOKEN"):
        return
    tok = Path.home() / ".cache/huggingface/token"
    if tok.is_file():
        os.environ["HF_TOKEN"] = tok.read_text().strip()
        log("HF_TOKEN set from ~/.cache/huggingface/token")
    else:
        log("WARNING: no HF_TOKEN and no cached token file; DINOv3 load may fail")


def ensure_val2014() -> None:
    """Download ONLY COCO val2014 (skip the 13 GB train2014 the repo downloader
    also pulls). Idempotent."""
    n = len(list(COCO_DIR.glob("*.jpg"))) if COCO_DIR.is_dir() else 0
    if n >= 40000:
        log(f"COCO val2014 present ({n} jpgs); skipping download")
        return
    parent = COCO_DIR.parent
    parent.mkdir(parents=True, exist_ok=True)
    zip_path = parent / "val2014.zip"
    url = "http://images.cocodataset.org/zips/val2014.zip"
    log(f"downloading val2014 (~6.3 GB) → {zip_path}")
    subprocess.run(["curl", "-L", "--retry", "3", "-C", "-", "-o", str(zip_path), url], check=True)
    log("unzipping val2014 …")
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(parent)], check=True)
    zip_path.unlink(missing_ok=True)
    n = len(list(COCO_DIR.glob("*.jpg")))
    log(f"val2014 ready ({n} jpgs)")


def model_entry(tag: str, models_cfg: dict) -> dict:
    """Return the models.yaml entry for `tag` with input_size overridden to 512
    (in-memory only; the yaml file is never modified)."""
    for group in ("baselines", "trained"):
        if tag in (models_cfg.get(group) or {}):
            entry = dict(models_cfg[group][tag])
            tfm = dict(entry.get("transform", {}) or {})
            tfm["input_size"] = INPUT_SIZE
            entry["transform"] = tfm
            return entry
    raise KeyError(f"tag {tag!r} not found in models.yaml baselines/trained")


def build_stems(n_global: int, write_csv: bool) -> list[str]:
    """Deterministic stem list of length n_global drawn from val2014 on disk.

    Pins the existing committed 2000 (`configs/coco_sample_ids.csv`) first for
    comparability, then seeded-samples the remainder. Only stems with a jpg on
    disk are used.
    """
    avail = sorted(p.stem for p in COCO_DIR.glob("*.jpg"))
    avail_set = set(avail)
    pinned_all = pd.read_csv(_CONFIGS / "coco_sample_ids.csv")["image_stem"].tolist()
    pinned = [s for s in pinned_all if s in avail_set]
    if n_global <= len(pinned):
        stems = pinned[:n_global]
    else:
        pool = [s for s in avail if s not in set(pinned)]
        rng = np.random.default_rng(SEED)
        idx = sorted(rng.choice(len(pool), size=n_global - len(pinned), replace=False))
        stems = pinned + [pool[i] for i in idx]
    assert len(stems) == n_global and len(set(stems)) == n_global, "stem list not unique/complete"
    if write_csv:
        out = _CROSS / "coco_sample_ids_5k.csv"
        pd.DataFrame({"image_stem": stems}).to_csv(out, index=False)
        log(f"wrote {out} ({len(stems)} stems)")
    return stems


# ══════════════════════════════════════════════════════════════════════════════
# Extraction (worker + driver fan-out)
# ══════════════════════════════════════════════════════════════════════════════
def run_worker(shard: int, num_shards: int, stems_file: Path) -> None:
    """Worker mode: extract dinov3+cd for this shard's stems (strided slice)."""
    ensure_hf_token()
    stems_all = stems_file.read_text().splitlines()
    stems = stems_all[shard::num_shards]
    models_cfg = yaml.safe_load((_CONFIGS / "models.yaml").read_text())
    log(f"worker shard {shard}/{num_shards}: {len(stems)} stems on "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    t0 = time.time()
    for tag in TAGS:
        extract_features(
            tag=tag,
            model_entry=model_entry(tag, models_cfg),
            image_stems=stems,
            coco_dir=COCO_DIR,
            features_root=FEATURES_ROOT,
            device="cuda",
        )
    log(f"worker shard {shard} done in {time.time() - t0:.0f}s")


def run_extraction(stems: list[str], num_shards: int = NUM_SHARDS) -> None:
    """Driver: write the stem list, spawn `num_shards` worker subprocesses
    (round-robin across GPUs), wait for all."""
    _RESULTS.mkdir(parents=True, exist_ok=True)
    stems_file = _RESULTS / "_stems.txt"
    stems_file.write_text("\n".join(stems))
    procs = []
    for w in range(num_shards):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(w % N_GPUS)
        cmd = [sys.executable, str(_THIS), "--worker",
               "--shard", str(w), "--num-shards", str(num_shards),
               "--stems-file", str(stems_file)]
        log(f"spawn worker {w} → GPU {w % N_GPUS}")
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    if any(rc != 0 for rc in rcs):
        raise RuntimeError(f"extraction workers failed: return codes {rcs}")
    log("all extraction workers finished")


# ══════════════════════════════════════════════════════════════════════════════
# Build the four source matrices (faithful to orchestrate.build_*_representations)
# ══════════════════════════════════════════════════════════════════════════════
def fit_cd_patch_pca(stems: list[str], fit_patch_sub: int) -> dict:
    """Fit ONE per-layer CD PCA basis (layers 2/5/8 separately) on CleanDIFT
    *patch tokens* sampled across `stems`. This single basis is applied per-patch
    for both the global and patch representations."""
    fr = str(FEATURES_ROOT)
    idx = make_patch_indices(n_images=len(stems), n_patches_total=TARGET * TARGET,
                             n_subsample=fit_patch_sub, seed=SEED)
    fit_layers = [build_patch_matrix_with_indices(fr, layer, stems, idx, target=TARGET)
                  for layer, _ in CD_LAYERS]
    # reducers=None → fits a separate PCAReducer per layer; we discard the
    # transformed output here and keep only the fitted bases.
    _, reducers = build_diffusion_combined_pca(
        *fit_layers, PCA_DIM, reducers=None, layer_keys=CD_LAYER_KEYS)
    comps = {k: reducers[k].actual_n_components for k in CD_LAYER_KEYS}
    log(f"CD PCA fit on patch tokens: {len(stems)}×{fit_patch_sub}="
        f"{len(stems) * fit_patch_sub} patches → components/layer {comps}")
    return reducers


def cd_patch_vectors(stems: list[str], patch_idx, reducers: dict) -> np.ndarray:
    """CD_p for sampled patches: per patch, transform each layer with the fitted
    basis → L2-norm per layer → concat. Returns (sum_i len(idx_i), 1536)."""
    fr = str(FEATURES_ROOT)
    layer_mats = [build_patch_matrix_with_indices(fr, layer, stems, patch_idx, target=TARGET)
                  for layer, _ in CD_LAYERS]
    cd_p, _ = build_diffusion_combined_pca(
        *layer_mats, PCA_DIM, reducers=reducers, layer_keys=CD_LAYER_KEYS)
    return cd_p


def cd_global_vectors(stems: list[str], reducers: dict) -> np.ndarray:
    """CD_p̄ per image: transform ALL patches per layer → L2-norm per layer →
    concat (= per-patch CD_p) → mean-pool over the 1024 patches. Streamed
    per-image to keep memory flat. Returns (len(stems), 1536)."""
    fr = str(FEATURES_ROOT)
    rows = []
    for stem in stems:
        parts = []
        for layer, key in CD_LAYERS:
            feat = load_feature(f"{fr}/{layer}/val/{stem}.npy")     # (H*W, C)
            feat = interpolate_to_32x32(feat, target=TARGET)         # (1024, C)
            parts.append(_row_normalize(reducers[key].transform(feat)))  # (1024, 512)
        cd_p = np.concatenate(parts, axis=1)                         # (1024, 1536)
        rows.append(cd_p.mean(axis=0))                               # (1536,)
    return np.stack(rows, axis=0)


def build_matrices(stems: list[str], n_patch_images: int, n_patch_sub: int,
                   fit_patch_sub: int, mode: str, rebuild: bool = False):
    # Cache the four matrices: the CD global streaming is the slow part (~12 min),
    # so alpha-grid re-runs reload instead of recomputing.
    cache = _RESULTS / f"_matrices_{mode}.npz"
    if cache.is_file() and not rebuild:
        z = np.load(cache)
        log(f"loaded cached matrices from {cache.name}")
        return z["DN_g"], z["CD_g"], z["DN_p"], z["CD_p"], z["groups"]

    fr = str(FEATURES_ROOT)

    # ---- 0. Fit the per-layer CD PCA basis on patch tokens (shared by both levels) ----
    reducers = fit_cd_patch_pca(stems, fit_patch_sub)
    exp_cd = sum(reducers[k].actual_n_components for k in CD_LAYER_KEYS)

    # ---- 1. Global (one vector per image) ----
    DN_g = build_global_matrix(fr, "dinov3_cls", stems, target=TARGET)          # (N, 1280) CLS
    CD_g = cd_global_vectors(stems, reducers)                                   # (N, 1536) mean of CD_p

    # ---- 2. Patch (one vector per sampled patch token) ----
    stems_p = stems[:n_patch_images]
    patch_idx = make_patch_indices(n_images=n_patch_images, n_patches_total=TARGET * TARGET,
                                   n_subsample=n_patch_sub, seed=SEED)
    DN_p = build_patch_matrix_with_indices(fr, "dinov3_patches", stems_p, patch_idx, target=TARGET)
    CD_p = cd_patch_vectors(stems_p, patch_idx, reducers)                       # same basis as CD_g
    groups = np.repeat(np.arange(n_patch_images), n_patch_sub)                  # image id per row

    # ---- Shape guards ----
    n_patch_rows = n_patch_images * n_patch_sub
    assert DN_g.shape == (len(stems), 1280), DN_g.shape
    assert CD_g.shape == (len(stems), exp_cd), (CD_g.shape, exp_cd)
    assert DN_p.shape == (n_patch_rows, 1280), DN_p.shape
    assert CD_p.shape == (n_patch_rows, exp_cd), (CD_p.shape, exp_cd)
    log(f"matrices: DN_g{DN_g.shape} CD_g{CD_g.shape} DN_p{DN_p.shape} CD_p{CD_p.shape} "
        f"(CD width={exp_cd}; =1536 in full)")
    DN_g, CD_g, DN_p, CD_p = (a.astype(np.float32) for a in (DN_g, CD_g, DN_p, CD_p))
    np.savez(cache, DN_g=DN_g, CD_g=CD_g, DN_p=DN_p, CD_p=CD_p, groups=groups)
    log(f"cached matrices → {cache.name}")
    return DN_g, CD_g, DN_p, CD_p, groups


# ══════════════════════════════════════════════════════════════════════════════
# Cross-prediction
# ══════════════════════════════════════════════════════════════════════════════
def cross_predict(X, Y, splitter, groups=None) -> dict:
    """Regularized linear regression X→Y under CV. Per-column standardization fit
    on the train fold only (inputs AND targets). Returns per-fold R² (multi-output
    uniform_average) + chosen alpha + sanity flags."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    n, d_in = X.shape
    d_out = Y.shape[1]
    r2s, alphas, n_trains = [], [], []
    split = splitter.split(X, Y, groups) if groups is not None else splitter.split(X)
    for tr, te in split:
        if groups is not None:  # leakage guard
            assert not (set(groups[tr]) & set(groups[te])), "GroupKFold image-id leak"
        xs, ys = StandardScaler(), StandardScaler()
        Xtr = xs.fit_transform(X[tr]); Xte = xs.transform(X[te])
        Ytr = ys.fit_transform(Y[tr]); Yte = ys.transform(Y[te])
        model = RidgeCV(alphas=ALPHAS).fit(Xtr, Ytr)
        pred = model.predict(Xte)
        r2s.append(float(r2_score(Yte, pred, multioutput="uniform_average")))
        alphas.append(float(np.atleast_1d(model.alpha_).mean()))
        n_trains.append(len(tr))
    n_train_min = int(min(n_trains))
    return dict(
        r2_per_fold=[round(v, 6) for v in r2s],
        r2_mean=round(float(np.mean(r2s)), 6),
        r2_std=round(float(np.std(r2s)), 6),
        alpha=round(float(np.mean(alphas)), 4),
        alpha_per_fold=[round(a, 4) for a in alphas],
        n=int(n), d=int(d_out), d_in=int(d_in),
        n_train_min=n_train_min,
        n_leq_d=bool(n_train_min <= d_out),
        n_train_over_d=round(n_train_min / d_out, 3),
        standardized=True,
    )


def run_analysis(matrices, mode: str) -> list[dict]:
    from sklearn.model_selection import KFold, GroupKFold
    DN_g, CD_g, DN_p, CD_p, groups = matrices
    records = []

    log("=== GLOBAL (KFold-5, shuffle, seed 42) ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for direction, X, Y in (("DN->CD", DN_g, CD_g), ("CD->DN", CD_g, DN_g)):
        r = cross_predict(X, Y, kf)
        r.update(setting="global", direction=direction, fold_grouping="KFold(5,shuffle,seed=42)")
        records.append(r)
        log(f"  {direction}: R²={r['r2_mean']:.4f}±{r['r2_std']:.4f} "
            f"alpha={r['alpha']} n={r['n']} d={r['d']} n_leq_d={r['n_leq_d']} "
            f"(n_train/d={r['n_train_over_d']})")

    log("=== PATCH (GroupKFold-5 on image id) ===")
    gkf = GroupKFold(n_splits=5)
    for direction, X, Y in (("DN->CD", DN_p, CD_p), ("CD->DN", CD_p, DN_p)):
        r = cross_predict(X, Y, gkf, groups=groups)
        r.update(setting="patch", direction=direction, fold_grouping="GroupKFold(5,image_id)")
        records.append(r)
        log(f"  {direction}: R²={r['r2_mean']:.4f}±{r['r2_std']:.4f} "
            f"alpha={r['alpha']} n={r['n']} d={r['d']} n_leq_d={r['n_leq_d']} "
            f"(n_train/d={r['n_train_over_d']})")
    return records


def write_outputs(records: list[dict], mode: str) -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if mode == "smoke" else ""
    cols = ["setting", "direction", "r2_mean", "r2_std", "r2_per_fold",
            "alpha", "alpha_per_fold", "n", "d", "d_in", "n_train_min",
            "n_leq_d", "n_train_over_d", "fold_grouping", "standardized"]
    df = pd.DataFrame(records)[cols]
    (jp := _RESULTS / f"cross_prediction{suffix}.json").write_text(json.dumps(records, indent=2))
    df.to_csv(_RESULTS / f"cross_prediction{suffix}.csv", index=False)
    log(f"wrote {jp} and .csv")

    # Summary table
    print("\n" + "=" * 64)
    print(f"  CROSS-PREDICTION SUMMARY  (mode={mode})")
    print("=" * 64)
    print(f"  {'setting':8} {'direction':8} {'R² mean±std':>16}   {'n':>7} {'d':>5}  n≤d")
    print("-" * 64)
    for r in records:
        print(f"  {r['setting']:8} {r['direction']:8} "
              f"{r['r2_mean']:.4f} ± {r['r2_std']:.4f}   "
              f"{r['n']:>7} {r['d']:>5}  {r['n_leq_d']}")
    print("=" * 64 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true", help="(internal) extract one shard")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    ap.add_argument("--stems-file", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--skip-extract", action="store_true",
                    help="features already cached; go straight to analysis")
    ap.add_argument("--rebuild-matrices", action="store_true",
                    help="ignore cached matrices npz and rebuild from features")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.shard, args.num_shards, Path(args.stems_file))
        return

    if args.smoke == args.full:
        raise SystemExit("specify exactly one of --smoke / --full")
    mode = "smoke" if args.smoke else "full"
    sz = SIZES[mode]
    log(f"mode={mode} sizes={sz}")
    log(f"FEATURES_ROOT={FEATURES_ROOT}")

    ensure_hf_token()
    ensure_val2014()
    stems = build_stems(sz["n_global"], write_csv=(mode == "full"))
    log(f"sample list: {len(stems)} stems")

    if not args.skip_extract:
        run_extraction(stems)

    matrices = build_matrices(stems, sz["n_patch_images"], sz["n_patch_sub"],
                              sz["fit_patch_sub"], mode=mode, rebuild=args.rebuild_matrices)
    records = run_analysis(matrices, mode)
    write_outputs(records, mode)


if __name__ == "__main__":
    main()
