"""vLLM offline captioner. Import this module lazily (inside a function), after
CUDA_VISIBLE_DEVICES has been finalized by the caller -- never at module load time.

API notes specific to vllm==0.8.5.post1 (pinned in env/recaption_env.txt):
  - LLM.chat() in this version does not accept raw PIL images in message content
    (only image_url / image_embeds), so we build the prompt ourselves via the HF
    processor's chat template and call LLM.generate() with multi_modal_data instead.
  - No separate image resize before this step: Gemma-3's processor resizes every
    image to a fixed 896x896 internally regardless of input size, so the image is
    passed through as decoded from disk.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class CaptionResult:
    text: str


class Captioner:
    def __init__(self, hf_id: str, tensor_parallel_size: int, gpu_memory_utilization: float,
                 max_model_len: int, max_num_seqs: int, prompt_text: str):
        from transformers import AutoProcessor
        from vllm import LLM

        self.llm = LLM(
            model=hf_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
        )
        self.processor = AutoProcessor.from_pretrained(hf_id)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }]
        self.prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def caption_batch(self, images: list[Image.Image], temperature: float, max_tokens: int,
                       seed: int) -> list[CaptionResult]:
        from vllm import SamplingParams

        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
        inputs = [{"prompt": self.prompt, "multi_modal_data": {"image": img}} for img in images]
        outputs = self.llm.generate(inputs, sp, use_tqdm=False)
        return [CaptionResult(text=o.outputs[0].text) for o in outputs]
