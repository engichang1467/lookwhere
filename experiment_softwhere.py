"""SoftWhere (P3) — multi-foveal selector visualization.

Renders, for one image, a row of panels:
  [ input | LookWhere single map | SoftWhere aggregate | TL map_1 ... TL map_S ]

LookWhere's selector produces ONE aggregated importance map; SoftWhere's
TokenLearner produces S distinct soft attention maps (multi-foveal) whose
aggregate is the importance map fed to top-k. This visual is the core
differentiator for the pitch.

One figure per TokenLearner variant is written:
  softwhere_v10.png  (sigmoid; lead with this — blobby/distinct even untrained)
  softwhere_v11.png  (softmax over space; flatter when untrained)

If a distilled head checkpoint is passed via --distilled, the head weights are
loaded so the maps are MEANINGFUL (see distill_decompose.py). Otherwise the
head is randomly initialized and the maps illustrate the *mechanism* (distinct
foveal regions), not learned saliency.

Run:  .venv/bin/python experiment_softwhere.py [--distilled softwhere_head_v10.pt]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modeling import LookWhereDownstream

# ----------------------------- settings -----------------------------
checkpoint = "lookwhere_dinov2.pt"
image_path = "ice_cream.jpg"
high_res_img_size = 518
k_ratio = 0.10
num_tokens = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--image", default=image_path)
parser.add_argument("--distilled", default=None,
                    help="path to a distilled TokenLearner head state_dict (.pt); "
                         "if given, the variant is inferred from the filename")
parser.add_argument("--variant", choices=["auto", "v10", "v11"], default="auto")
parser.add_argument("--num-tokens", type=int, default=num_tokens)
parser.add_argument("--tl-agg", choices=["max", "mean", "logsumexp"], default="max")
parser.add_argument("--tl-sr-mode", choices=["none", "conv"], default="none")
args = parser.parse_args()

num_patches = (high_res_img_size // 14) ** 2
k = int(k_ratio * num_patches)
grid = high_res_img_size // 14

transform = transforms.Compose([
    transforms.Resize(high_res_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(high_res_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                         std=torch.tensor([0.229, 0.224, 0.225])),
])
pil = Image.open(args.image).convert("RGB")
image = transform(pil).unsqueeze(0).to(device)
base = np.array(pil.resize((high_res_img_size, high_res_img_size)))


def norm01(a):
    a = a.astype(np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def show_overlay(ax, saliency_2d, title):
    """Image with a red saliency heatmap overlaid (alpha)."""
    ax.imshow(base)
    s = norm01(saliency_2d)
    big = np.array(Image.fromarray((s * 255).astype(np.uint8)).resize(
        (high_res_img_size, high_res_img_size), Image.NEAREST)) / 255.0
    ax.imshow(big, cmap="inferno", alpha=0.55)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# --- LookWhere single map (pretrained MLP head) ---
lw_mlp = LookWhereDownstream(checkpoint, high_res_size=high_res_img_size, num_classes=0,
                             k=k, is_cls=True, device=device, head_type="mlp")
lw_mlp.eval()
with torch.no_grad():
    lw_map = lw_mlp.selector(image)["selector_map"].reshape(grid, grid).float().cpu().numpy()


def render(variant):
    lw_tl = LookWhereDownstream(checkpoint, high_res_size=high_res_img_size, num_classes=0,
                                k=k, is_cls=True, device=device, head_type="tokenlearner",
                                num_tokens=args.num_tokens, tl_variant=variant,
                                tl_agg=args.tl_agg, tl_sr_mode=args.tl_sr_mode)
    tag = "untrained"
    if args.distilled:
        sd = torch.load(args.distilled, map_location=device, weights_only=True)
        lw_tl.selector.head.load_state_dict(sd)
        tag = "distilled"
    lw_tl.eval()
    with torch.no_grad():
        sel = lw_tl.selector(image)
        agg = sel["selector_map"].reshape(grid, grid).float().cpu().numpy()
        attn_t = getattr(lw_tl.selector.head, "_last_attn_sr", None)
        if attn_t is None:
            attn_t = lw_tl.selector.head._last_attn
        attn = attn_t[0].float().cpu().numpy()
    S = attn.shape[0]

    ncols = 3 + S
    fig, axes = plt.subplots(1, ncols, figsize=(2.4 * ncols, 2.7))
    axes[0].imshow(base); axes[0].set_title("input", fontsize=9); axes[0].axis("off")
    show_overlay(axes[1], lw_map, "LookWhere\n(single map)")
    show_overlay(axes[2], agg, f"SoftWhere agg\n({variant}, {tag})")
    for s in range(S):
        show_overlay(axes[3 + s], attn[s], f"foveal map {s + 1}")
    fig.suptitle(f"SoftWhere multi-foveal selector — {variant} ({tag})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    sr_tag = "_sr" if args.tl_sr_mode == "conv" else ""
    out_path = f"softwhere_{variant}{sr_tag}_{tag}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}  (input + LookWhere + agg + {S} foveal maps)")


variants = ["v10", "v11"]
if args.variant != "auto":
    variants = [args.variant]
elif args.distilled:  # infer single variant from filename to load matching weights
    variants = ["v11" if "v11" in args.distilled else "v10"]
for v in variants:
    render(v)
