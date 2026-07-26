"""Two-stream single-forward Contrastive Region Guidance (CRG) decode.

CRG combines a *positive* stream (the full image) with a *negative* stream (the same
image with the queried region blacked out) at the logit level::

    logits = (1 + alpha) * logit_pos - alpha * logit_neg

For a multiple-choice perception question we do NOT run a generation loop: the prompt
ends with a format instruction so the model's FIRST answer token is the decision (e.g.
``top`` / ``bottom``). We read the next-token logits at the generation-prompt
position, restrict them to the option-token ids, apply the CRG combine, and softmax
over the options. One forward pass per batch, and the per-option probabilities give
the AUROC signal directly.

Model-agnostic across the VLM families in this experiment (Qwen2.5-VL, Qwen3.x,
Gemma-3/4, InternVL3). Image handling branches on the family, which is declared per
model as ``family`` in ``configs/models.yaml`` rather than sniffed from the id.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_ALPHA = 1.0


@dataclass
class _Loaded:
    model: object
    proc: object
    device: str
    family: str          # "qwen" -> needs qwen_vl_utils image extraction; else PIL direct
    no_think: bool       # pass enable_thinking=False so the first token is the answer


_STATE: _Loaded | None = None


def _state() -> _Loaded:
    if _STATE is None:
        raise RuntimeError(
            "decode engine not loaded — call load(model_id=..., family=...) first")
    return _STATE


def load(model_id: str, *, family: str = "hf", no_think: bool = False, dtype=None,
         load_4bit: bool = False, max_memory: dict | None = None) -> _Loaded:
    """Load (model, processor, device) once and cache it.

    ``model_id`` is required: there is no default, because silently probing whichever
    model happened to be hardcoded would produce results attributed to the wrong one.

    load_4bit: bitsandbytes nf4 (bf16 compute). The vision tower, multimodal connector
      and lm_head stay unquantized — their norms cannot run on the 4-bit storage dtype.
      Needed to fit the largest models on the available GPUs.
    max_memory: e.g. {0: '42GiB', 1: '42GiB'} -> device_map='auto' splits across GPUs
      within those caps; otherwise the model loads onto a single GPU.
    """
    global _STATE
    if _STATE is not None:
        return _STATE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_awq = "awq" in model_id.lower()
    kw: dict = dict(attn_implementation="sdpa")
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # keep vision tower + multimodal connector + lm_head unquantized
            # (Qwen: 'visual'; InternVL: 'vision_tower'/'multi_modal_projector'/'mlp1')
            llm_int8_skip_modules=["visual", "vision_tower",
                                   "multi_modal_projector", "mlp1", "lm_head"])
    else:
        kw["dtype"] = dtype or (torch.float16 if is_awq else torch.bfloat16)
    kw["device_map"] = "auto" if max_memory else {"": 0}
    if max_memory:
        kw["max_memory"] = max_memory

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    # LEFT padding keeps the LAST position a real (answer-predicting) token in every
    # row of a batch whose prompts differ in length.
    proc.tokenizer.padding_side = "left"
    _STATE = _Loaded(model=model, proc=proc, device=device, family=family,
                     no_think=no_think)
    return _STATE


def option_token_ids(options: list[str]) -> list[int]:
    """First-token id of each option string (the decision token to read)."""
    tok = _state().proc.tokenizer
    return [tok.encode(str(o), add_special_tokens=False)[0] for o in options]


def _proc_call(st: _Loaded, texts: list[str], convs: list, images: list):
    """Build processor tensors, branching on the model family.

    Qwen-VL needs ``qwen_vl_utils.process_vision_info`` to extract image inputs;
    Gemma/InternVL take PIL images directly via the processor (nested one list per
    text sample), which expands the chat-template image placeholder.
    """
    if st.family == "qwen":
        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(convs)
        return st.proc(text=texts, images=image_inputs, padding=True, return_tensors="pt")
    return st.proc(text=texts, images=[[im] for im in images],
                   padding=True, return_tensors="pt")


def _tmpl_kw(st: _Loaded) -> dict:
    ct = st.proc.tokenizer.chat_template or ""
    return {"enable_thinking": False} if (st.no_think and "enable_thinking" in ct) else {}


def _build_inputs(st: _Loaded, images: list, prompts) -> dict:
    """Process images + (shared str or per-image list of) prompts into one batch."""
    prompts = [prompts] * len(images) if isinstance(prompts, str) else list(prompts)
    assert len(prompts) == len(images), "prompts and images must align"
    convs = [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": p},
    ]}] for img, p in zip(images, prompts)]
    texts = [st.proc.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                         **_tmpl_kw(st))
             for c in convs]
    inputs = _proc_call(st, texts, convs, images)
    return {k: v.to(st.device) for k, v in inputs.items()}


@torch.inference_mode()
def decision_batch(pos_images: list, prompts, option_ids: list[int],
                   neg_images: list | None = None, alpha: float = DEFAULT_ALPHA):
    """Single-forward CRG decision over a fixed option-token set (no generation).

    Args:
        pos_images: B positive (full) images.
        prompts: one shared prompt str, or a list of B per-image prompts.
        option_ids: token ids of the answer options (from ``option_token_ids``).
        neg_images: B negative (region-blacked) images. ``None`` => raw (no CRG).
        alpha: CRG guidance strength.

    Returns:
        (probs, preds): probs is a list of B per-option probability lists; preds is
        the list of B argmax option indices.
    """
    st = _state()
    crg = neg_images is not None
    B = len(pos_images)
    imgs = list(pos_images) + (list(neg_images) if crg else [])
    pr = prompts if isinstance(prompts, str) else (
        list(prompts) + (list(prompts) if crg else []))
    inputs = _build_inputs(st, imgs, pr)
    logits = st.model(**inputs, return_dict=True).logits[:, -1, :].float()   # (N, V)
    comb = ((1.0 + alpha) * logits[:B] - alpha * logits[B:]) if crg else logits
    probs = torch.softmax(comb[:, option_ids], dim=-1)                       # (B, n_opts)
    return probs.cpu().tolist(), probs.argmax(dim=-1).cpu().tolist()


def free_model() -> None:
    """Release the cached model + free CUDA memory (between models in a sweep)."""
    global _STATE
    import gc
    _STATE = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
