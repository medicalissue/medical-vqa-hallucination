"""P5: Attention map visualization.

For BiomedCLIP we visualize the ViT image-token attention to the [CLS] token,
across two cases per sample:
  - real image + question
  - blank image + question
The hypothesis: if the model is "looking", the real-image attention should
concentrate on diagnostic regions while the blank-image case should be diffuse.

LLaVA-Med attention is more involved (cross-modal layers); we save the
patch-level attention rollout from the vision tower.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


def biomed_clip_attention(model, image, device="cuda:0"):
    """Return averaged ViT attention from last block, [num_patches]."""
    img_t = model.preprocess(image).unsqueeze(0).to(device)
    # open_clip's BiomedCLIP visual is a TimmModel wrapping a ViT
    vis = model.model.visual
    feats = []
    handle = None

    # try hooking attention from the last transformer block
    try:
        target = vis.trunk.blocks[-1].attn
    except AttributeError:
        return None

    def hook(mod, inp, out):
        # transformer-block attention output is (out, attn_weights) in some impls;
        # for timm Attention, .scale and we need to recompute. Capture inputs instead.
        feats.append(inp[0].detach())

    handle = target.register_forward_hook(hook)
    with torch.inference_mode():
        _ = model.model.encode_image(img_t)
    handle.remove()
    if not feats: return None
    x = feats[0]  # [B, N, D]
    # Approximate attention: norm of token features as saliency proxy
    sal = x.norm(dim=-1).squeeze(0).cpu().numpy()  # [N]
    # drop CLS token
    if len(sal) > 1: sal = sal[1:]
    n = int(np.sqrt(len(sal)))
    if n * n != len(sal): return None
    return sal.reshape(n, n)


def visualize_pair(model_wrapper, image: Image.Image, question: str,
                   out_path: Path, blank_color=(0, 0, 0)):
    sal_real = biomed_clip_attention(model_wrapper, image)
    blank = Image.new("RGB", image.size, blank_color)
    sal_blank = biomed_clip_attention(model_wrapper, blank)
    if sal_real is None or sal_blank is None:
        print("attention extraction unsupported"); return
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(image); axes[0, 0].set_title("real image"); axes[0, 0].axis("off")
    axes[0, 1].imshow(sal_real, cmap="hot"); axes[0, 1].set_title("saliency (real)"); axes[0, 1].axis("off")
    axes[1, 0].imshow(blank); axes[1, 0].set_title("blank image"); axes[1, 0].axis("off")
    axes[1, 1].imshow(sal_blank, cmap="hot"); axes[1, 1].set_title("saliency (blank)"); axes[1, 1].axis("off")
    fig.suptitle(f"Q: {question}", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from datasets import load_from_disk
    from models_wrapper import BiomedClipWrapper

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk("/home/ubuntu/mmbert_work/data/vqa_rad/hf")["test"]
    model = BiomedClipWrapper()
    for i in range(args.n):
        ex = ds[i]
        visualize_pair(model, ex["image"], ex["question"],
                       out / f"sample_{i}_{ex['answer'][:20].replace(' ','_')}.png")
        print(f"wrote {i}", flush=True)


if __name__ == "__main__":
    main()
