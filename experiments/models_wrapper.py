"""Uniform inference wrappers for the two medical VQA models.

Every model exposes:
    model.answer(image, question) -> {"answer": str, "raw": str | None}
Optional:
    model.answer_with_scores(image, question, answer_candidates)
        -> {"scores": list[float], "probs": list[float]}

This keeps the experiment driver model-agnostic.
"""
from __future__ import annotations
import os
from typing import List, Dict, Optional
import torch
from PIL import Image


class LlavaMedWrapper:
    """Generative VQA via LLaVA-Med v1.5 (HF-compatible fork)."""

    MODEL_ID = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

    def __init__(self, device="cuda:0", dtype=torch.float16):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=device
        )
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def answer(self, image: Image.Image, question: str, max_new_tokens: int = 64) -> Dict:
        prompt = f"USER: <image>\n{question} ASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=self.processor.tokenizer.eos_token_id)
        full = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        ans = full.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in full else full
        return {"answer": ans, "raw": full}


class BiomedClipWrapper:
    """Contrastive zero-shot VQA via BiomedCLIP. Given an image+question, we
    score a candidate answer set and return argmax + probability (softmax over
    image-text similarities). The candidate set is supplied per call.
    """

    MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self, device="cuda:0"):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL_ID)
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_ID)
        self.model.to(device).eval()

    @torch.inference_mode()
    def answer_with_scores(self, image: Image.Image, question: str,
                            candidates: List[str]) -> Dict:
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        texts = [f"{question} {c}" for c in candidates]
        tok = self.tokenizer(texts).to(self.device)
        img_feat = self.model.encode_image(img)
        txt_feat = self.model.encode_text(tok)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        logits = (img_feat @ txt_feat.T).squeeze(0)  # [C]
        probs = torch.softmax(logits * 100.0, dim=-1).cpu().tolist()
        best = int(logits.argmax().item())
        return {
            "answer": candidates[best],
            "candidates": candidates,
            "probs": probs,
            "top_prob": probs[best],
        }

    def answer(self, image: Image.Image, question: str,
               candidates: Optional[List[str]] = None) -> Dict:
        if candidates is None:
            # Default yes/no candidate set; caller supplies real set in driver.
            candidates = ["yes", "no"]
        r = self.answer_with_scores(image, question, candidates)
        return {"answer": r["answer"], "raw": None, "confidence": r["top_prob"]}


MODEL_REGISTRY = {
    "llava_med": LlavaMedWrapper,
    "biomed_clip": BiomedClipWrapper,
}
