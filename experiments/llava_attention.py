"""LLaVA-Med attention rollout from the CLIP vision tower.

We grab the last self-attention map from the CLIP-ViT-L vision encoder, average
over heads, optionally apply attention rollout (Abnar & Zuidema, 2020) across
all layers. We then compare real vs blank image attention for the same sample.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


def attention_rollout(attns_per_layer):
    """attns_per_layer: list[L] of [B, heads, T, T] tensors. Returns [T] for CLS attn."""
    res = None
    for a in attns_per_layer:
        a = a.mean(dim=1)              # [B, T, T] (avg heads)
        a = a + torch.eye(a.size(-1), device=a.device).unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True)
        res = a if res is None else torch.matmul(a, res)
    # CLS row
    cls = res[:, 0, 1:]  # drop CLS->CLS, keep CLS->patches
    return cls.squeeze(0).cpu().numpy()


def get_vision_attention(model, processor, image, device="cuda:0", dtype=torch.float16):
    """Run only the vision tower with output_attentions=True."""
    inputs = processor(text="USER: <image>\n? ASSISTANT:", images=image, return_tensors="pt").to(device)
    pix = inputs["pixel_values"].to(dtype)
    vision = model.vision_tower
    with torch.inference_mode():
        out = vision(pix, output_attentions=True)
    attns = out.attentions  # tuple of [B, heads, T, T]
    return attention_rollout(list(attns))  # [T-1]


def visualize(model_w, image: Image.Image, question: str, out_path: Path):
    sal_real = get_vision_attention(model_w.model, model_w.processor, image)
    blank = Image.new("RGB", image.size, (0, 0, 0))
    sal_blank = get_vision_attention(model_w.model, model_w.processor, blank)
    n = int(np.sqrt(len(sal_real)))
    if n * n != len(sal_real):
        # CLIP-ViT-L 224/14 patches = 16x16 = 256
        return
    sal_real = sal_real.reshape(n, n)
    sal_blank = sal_blank.reshape(n, n)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(image); axes[0, 0].set_title("real image"); axes[0, 0].axis("off")
    axes[0, 1].imshow(sal_real, cmap="hot"); axes[0, 1].set_title("attn rollout (real)"); axes[0, 1].axis("off")
    axes[1, 0].imshow(blank); axes[1, 0].set_title("blank image"); axes[1, 0].axis("off")
    axes[1, 1].imshow(sal_blank, cmap="hot"); axes[1, 1].set_title("attn rollout (blank)"); axes[1, 1].axis("off")
    fig.suptitle(f"Q: {question}", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from datasets import load_from_disk
    from models_wrapper import LlavaMedWrapper

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk("/home/ubuntu/mmbert_work/data/vqa_rad/hf")["test"]
    model_w = LlavaMedWrapper()
    for i in range(args.n):
        ex = ds[i]
        try:
            visualize(model_w, ex["image"], ex["question"],
                      out / f"sample_{i}_{str(ex['answer'])[:20].replace(' ','_')}.png")
            print(f"wrote {i}", flush=True)
        except Exception as e:
            print(f"sample {i} failed: {e}", flush=True)


if __name__ == "__main__":
    main()
