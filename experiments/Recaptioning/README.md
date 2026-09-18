# Recaptioning

Builds several seeded versions of a recaptioned image–text dataset: images come from
[`guangyil/laion-coco-aesthetic`](https://huggingface.co/datasets/guangyil/laion-coco-aesthetic),
captions are regenerated with a Gemma-3 VLM served by vLLM, and the result is split into versions
that share no images and differ only by sampling seed — so you can measure how much a downstream
result depends on which images you happened to draw.

## Setup

Needs **Python 3.11 or 3.12** — vLLM 0.8.5.post1 doesn't support 3.13+. 

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set these before the first run:

| Variable | Required | What it does |
|---|---|---|
| `HF_TOKEN` | **yes** | Gemma-3 weights are gated. Accept the licence on the [model page](https://huggingface.co/google/gemma-3-27b-it), then make a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). |
| `HF_HOME` | recommended | Where model weights and the source dataset are cached. Gemma-3-27B is ~54 GB and the source parquet ~4 GB — point it at a disk with room, or it fills your home directory. |
| `RECAPTION_OUT` | no | Default for `--out`, so you can omit the flag. |

```bash
export HF_TOKEN=hf_...
export HF_HOME=/path/to/big-disk/hf-cache     # any directory with ~60 GB free
```

**Hardware.** NVIDIA GPU, driver >= 550 (CUDA 12.4). `gemma-3-27b-it` needs ~80 GB of VRAM (one
H100) or two smaller cards with `--tensor-parallel 2`. `gemma-3-4b-it` fits in 20 GB and is the
right choice for a smoke test.

**Disk.** The smoke test needs a few hundred MB. A full run writes about **600 GB** to `--out`
(~290 GB of downloaded images, plus ~290 GB once they're packed into the four versions), so point
`--out` at a data disk rather than letting it default into the repo.

## Run

```bash
# smoke test first — 4 tiny versions, ~5 min, writes to ./recaption-data
python run.py all --per-seed 300 --model google/gemma-3-4b-it

# the real thing — 4 versions of 508,025 images each, ~600 GB
python run.py all --out /path/to/big-disk/recaption
```

`/path/to/big-disk` is just any directory you have space in — there's nothing special about it.
Every flag has a default, so the smoke test above needs no `--out` at all. Re-running the same
command resumes: downloaded images and finished caption chunks are skipped.

| Arg | Default | Meaning |
|---|---|---|
| `--out` | `$RECAPTION_OUT`, else `./recaption-data` | where everything is written; needs ~600 GB for a full run |
| `--seeds` | `4` | how many versions to produce |
| `--per-seed` | `508025` | images per version; versions share no images |
| `--model` | `google/gemma-3-27b-it` | any model id vLLM can serve |
| `--tensor-parallel` | `1` | GPUs per model replica; raise only if the weights don't fit |

Each stage prints live progress and a `done` marker; verbose vLLM and downloader logs go to
`<out>/logs/`.

```
[2/3] caption
  caption   rank0 1500/1904 (79%)   8.8 img/s   eta 0.8min
  caption done — rank0: 1904 captions
```

## More GPUs, more nodes

`caption` reads `RANK` / `WORLD_SIZE` / `LOCAL_RANK` from the environment, so `deepspeed`,
`torchrun` and `srun` all work. `download` and `assemble` are single-process — run them once.

Here `--out` must be a path **both nodes can see** (NFS, Lustre, any shared mount), because every
rank reads and writes the same `captions/` directory — that's how they split the work and resume.

```bash
python run.py download --out /shared/recaption

# on each node, changing only --node_rank
deepspeed --no_ssh --node_rank 0 --num_nodes 2 --num_gpus 8 --master_addr <node0-ip> \
    run.py caption --out /shared/recaption

python run.py assemble --out /shared/recaption
```

Captioning 2M images on 16 H100s takes roughly 3.5 h; downloading them takes longer and is
bandwidth-bound.

## Output

```
<out>/
├── images/              downloaded jpgs + index.jsonl (key, url, original_caption)
├── captions/            one jsonl per chunk (also the resume state)
└── versions/seed{0..3}/
    ├── metadata.csv     image_id,url,caption,original_caption
    ├── shards-*.tar     <image_id>.jpg + <image_id>.txt  (WebDataset layout)
    └── provenance.json  seed, model, prompt, counts, per-shard sha256
```

Tuning beyond the flags above — download concurrency, timeout, image size, the caption prompt —
lives in [`config.yaml`](config.yaml).
