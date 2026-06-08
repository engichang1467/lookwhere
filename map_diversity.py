"""SoftWhere (P3) — map-diversity vs teacher-fidelity tradeoff.

Quantifies the §6 collapse risk: are the S foveal maps actually distinct, or do
they collapse onto one blob? Sweeps the distilled heads produced by
distill_decompose.py at several diversity-regularizer weights and reports, per
weight:
  - fidelity   = KL(teacher || SoftWhere aggregate)  [lower = closer to teacher]
  - overlap    = mean pairwise histogram-intersection of the S maps  [lower = distinct]
  - diversity  = 1 - overlap                                          [higher = multi-foveal]
  - agreement  = recall of SoftWhere top-k vs teacher top-k           [the proxy metric]

The story: a small diversity weight buys distinct multi-foveal maps at little
fidelity cost; too much collapses fidelity. Auto-discovers
softwhere_head_<variant>_div*.pt and parses the weight from the filename.

Run:  .venv/bin/python map_diversity.py --variant v10
"""
import argparse
import glob
import os
import re

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from modeling import LookWhereDownstream

checkpoint = "lookwhere_dinov2.pt"
high_res_img_size = 518
k_ratio = 0.10
imagenette_val = os.path.join(
    os.environ.get("OTL_PATH", "/home/michael/ProjectE2/OpenTokenLearner"),
    "data/imagenette2-320/val")
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="v10", choices=["v10", "v11"])
parser.add_argument("--num_tokens", type=int, default=4)
parser.add_argument("--n_images", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

grid = high_res_img_size // 14
k = int(k_ratio * grid * grid)

transform = transforms.Compose([
    transforms.Resize(high_res_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(high_res_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                         std=torch.tensor([0.229, 0.224, 0.225])),
])
paths = sorted(glob.glob(os.path.join(imagenette_val, "*", "*.JPEG")))[: args.n_images]
imgs = [transform(Image.open(p).convert("RGB")) for p in paths]


def batches():
    for i in range(0, len(imgs), args.batch_size):
        yield torch.stack(imgs[i:i + args.batch_size]).to(device)


def topk_mask(m):
    idx = torch.topk(m, k=k, dim=-1).indices
    return torch.zeros_like(m).scatter_(1, idx, 1.0).bool()


def pairwise_overlap(attn):
    """Mean off-diagonal histogram-intersection of the S maps. attn: (B,S,g,g)."""
    b, s, h, w = attn.shape
    p = attn.reshape(b, s, h * w)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)          # per-map distribution
    inter = torch.minimum(p.unsqueeze(2), p.unsqueeze(1)).sum(-1)  # (B,S,S) in [0,1]
    eye = torch.eye(s, device=attn.device).bool()
    return inter[:, ~eye].mean().item()


# teacher saliency (pretrained MLP head), cached per image
lw_mlp = LookWhereDownstream(checkpoint, high_res_size=high_res_img_size, num_classes=0,
                             k=k, is_cls=True, device=device, head_type="mlp")
lw_mlp.eval()
teacher_maps, teacher_masks = [], []
with torch.no_grad():
    for b in batches():
        m = lw_mlp.selector(b)["selector_map"]
        teacher_maps.append(m)
        teacher_masks.append(topk_mask(m))
teacher_maps = torch.cat(teacher_maps)
teacher_masks = torch.cat(teacher_masks)
t_dist = F.softmax(teacher_maps, dim=-1)
del lw_mlp

heads = sorted(glob.glob(f"softwhere_head_{args.variant}_div*.pt"),
               key=lambda p: float(re.search(r"div([0-9]+(?:\.[0-9]+)?)", p).group(1)))
if not heads:
    raise SystemExit(f"no distilled heads found for variant={args.variant}; "
                     f"run distill_decompose.py first")

lw_tl = LookWhereDownstream(checkpoint, high_res_size=high_res_img_size, num_classes=0,
                            k=k, is_cls=True, device=device, head_type="tokenlearner",
                            num_tokens=args.num_tokens, tl_variant=args.variant, tl_agg="max")
lw_tl.eval()

print(f"\nvariant={args.variant}, S={args.num_tokens}, {len(imgs)} images, k={k}\n")
print(f"{'div_w':>6} {'fidelity_KL':>12} {'overlap':>9} {'diversity':>10} {'agreement':>10}")
print("-" * 52)
for hp in heads:
    w = float(re.search(r"div([0-9]+(?:\.[0-9]+)?)", hp).group(1))
    lw_tl.selector.head.load_state_dict(torch.load(hp, map_location=device, weights_only=True))
    kls, overlaps, recalls, n = [], [], [], 0
    with torch.no_grad():
        for bi, b in enumerate(batches()):
            sel = lw_tl.selector(b)
            s_logdist = F.log_softmax(sel["selector_map"], dim=-1)
            sl = slice(bi * args.batch_size, bi * args.batch_size + b.shape[0])
            kls.append(F.kl_div(s_logdist, t_dist[sl], reduction="batchmean").item() * b.shape[0])
            overlaps.append(pairwise_overlap(lw_tl.selector.head._last_attn) * b.shape[0])
            s_mask = topk_mask(sel["selector_map"])
            recalls.append((s_mask & teacher_masks[sl]).sum(-1).float().div(k).sum().item())
            n += b.shape[0]
    fid = sum(kls) / n
    ov = sum(overlaps) / n
    rec = sum(recalls) / n
    print(f"{w:>6g} {fid:>12.4f} {ov:>9.3f} {1 - ov:>10.3f} {rec:>10.3f}")
print("\noverlap: mean pairwise histogram-intersection of the S maps (1=identical, 0=disjoint)")
