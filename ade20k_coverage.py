"""SoftWhere (P3) — real multi-object coverage on ADE20K val.

The proposal's standalone claim (§4, §7): multi-foveal selection covers more
distinct objects than LookWhere's single aggregated map, on multi-object images.
This is the one result that holds even if downstream accuracy ties.

Metric — object-level recall via connected components:
  1. rasterize the GT semantic mask to the 37x37 patch grid (majority class/patch).
  2. connected components (4-conn) of same-class patches = distinct object regions;
     keep components with >= --min_obj_patches patches.
  3. evaluate only multi-object images (>= --min_objects kept components).
  4. a component is "covered" if >=1 of its patches is in the selected top-k.
  5. object-recall = covered components / total components, averaged over images.

Four selectors compared at k_ratio:
  - LookWhere single      : top-k of the pretrained MLP aggregate map (baseline)
  - SoftWhere aggregate   : top-k of the distilled TokenLearner aggregate (sanity:
                            distilled to match the teacher, so should ~= single)
  - SoftWhere multi-foveal: top-(k/S) from EACH of the S maps, unioned (the actual
                            multi-foveal selection the proposal claims helps)
  - random                : lower bound

Run:  .venv/bin/python ade20k_coverage.py --distilled softwhere_head_v10_div1.pt --variant v10
"""
import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision import transforms

from modeling import LookWhereDownstream

checkpoint = "lookwhere_dinov2.pt"
high_res_img_size = 518
k_ratio = 0.10
ade_root = "ade_data/ADEChallengeData2016"
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="v10", choices=["v10", "v11"])
parser.add_argument("--num_tokens", type=int, default=4)
parser.add_argument("--distilled", required=True)
parser.add_argument("--n_images", type=int, default=500)
parser.add_argument("--min_obj_patches", type=int, default=3)
parser.add_argument("--max_obj_patches", type=int, default=25,
                    help="upper size bound: focus on genuine small objects, not "
                         "whole-scene stuff regions (wall/floor/sky) that any 10%% "
                         "selection trivially covers")
parser.add_argument("--min_objects", type=int, default=3)
args = parser.parse_args()

grid = high_res_img_size // 14          # 37
num_patches = grid * grid
k = int(k_ratio * num_patches)
S = args.num_tokens

img_tf = transforms.Compose([
    transforms.Resize(high_res_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(high_res_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                         std=torch.tensor([0.229, 0.224, 0.225])),
])


def mask_to_patch_labels(mask_pil):
    """Center-crop-resize the mask the same way as the image, then majority-vote
    each 14x14 patch -> (grid, grid) int label array (0 = unlabeled)."""
    # replicate Resize(short side)->CenterCrop on the label map with NEAREST
    m = transforms.functional.resize(mask_pil, high_res_img_size,
                                     interpolation=transforms.InterpolationMode.NEAREST)
    m = transforms.functional.center_crop(m, high_res_img_size)
    arr = np.array(m)                                   # (518,518)
    arr = arr.reshape(grid, 14, grid, 14)
    # majority class per patch
    out = np.zeros((grid, grid), dtype=np.int64)
    for i in range(grid):
        for j in range(grid):
            vals, cnts = np.unique(arr[i, :, j, :], return_counts=True)
            out[i, j] = vals[cnts.argmax()]
    return out


def connected_components(labels):
    """4-connected components of equal (nonzero) label. Returns list of patch-index sets."""
    comps = []
    for cls in np.unique(labels):
        if cls == 0:
            continue
        lab, n = ndimage.label(labels == cls)   # default cross structure = 4-connectivity
        for c in range(1, n + 1):
            idxs = np.flatnonzero(lab.ravel() == c)
            if args.min_obj_patches <= len(idxs) <= args.max_obj_patches:
                comps.append(set(idxs.tolist()))
    return comps


def topk_set(scores_1d, kk):
    return set(torch.topk(scores_1d, k=kk).indices.tolist())


def multifoveal_set(maps_s):  # maps_s: (S, num_patches)
    per = max(1, k // S)
    sel = set()
    for s in range(S):
        sel |= topk_set(maps_s[s], per)
    # top up to k from the aggregate if rounding left us short
    if len(sel) < k:
        agg = maps_s.amax(0)
        for idx in torch.topk(agg, k=num_patches).indices.tolist():
            if idx not in sel:
                sel.add(idx)
            if len(sel) >= k:
                break
    return sel


def object_recall(comps, selected):
    if not comps:
        return None
    covered = sum(1 for c in comps if c & selected)
    return covered / len(comps)


# models
lw_mlp = LookWhereDownstream(checkpoint, high_res_size=high_res_img_size, num_classes=0,
                             k=k, is_cls=True, device=device, head_type="mlp")
lw_mlp.eval()
lw_tl = LookWhereDownstream(checkpoint, high_res_size=high_res_img_size, num_classes=0,
                            k=k, is_cls=True, device=device, head_type="tokenlearner",
                            num_tokens=S, tl_variant=args.variant, tl_agg="max")
lw_tl.selector.head.load_state_dict(torch.load(args.distilled, map_location=device, weights_only=True))
lw_tl.eval()

img_paths = sorted(glob.glob(os.path.join(ade_root, "images/validation/*.jpg")))[: args.n_images]
print(f"ADE20K val: {len(img_paths)} images, k={k}/{num_patches}, S={S}, "
      f"multi-object = >={args.min_objects} small objects of "
      f"{args.min_obj_patches}-{args.max_obj_patches} patches")

acc = {"single": [], "agg": [], "multifoveal": [], "random": []}
n_multi = 0
rng = torch.Generator(device=device).manual_seed(0)
with torch.no_grad():
    for p in img_paths:
        mask_p = os.path.join(ade_root, "annotations/validation",
                              os.path.basename(p).replace(".jpg", ".png"))
        labels = mask_to_patch_labels(Image.open(mask_p))
        comps = connected_components(labels)
        if len(comps) < args.min_objects:
            continue
        n_multi += 1

        x = img_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        single = lw_mlp.selector(x)["selector_map"][0]
        agg = lw_tl.selector(x)["selector_map"][0]
        # per-map scores at 37x37
        attn = lw_tl.selector.head._last_attn          # (1,S,11,11)
        maps_s = F.interpolate(attn, size=(grid, grid), mode="bilinear",
                               align_corners=False)[0].reshape(S, num_patches)

        acc["single"].append(object_recall(comps, topk_set(single, k)))
        acc["agg"].append(object_recall(comps, topk_set(agg, k)))
        acc["multifoveal"].append(object_recall(comps, multifoveal_set(maps_s)))
        rand = torch.rand(num_patches, generator=rng, device=device)
        acc["random"].append(object_recall(comps, topk_set(rand, k)))

print(f"\nmulti-object images evaluated: {n_multi}/{len(img_paths)}")
print("\n--- object-level coverage recall (higher = covers more distinct objects) ---")
for name in ["single", "agg", "multifoveal", "random"]:
    v = np.mean(acc[name])
    label = {"single": "LookWhere single map", "agg": "SoftWhere aggregate",
             "multifoveal": "SoftWhere multi-foveal", "random": "random"}[name]
    print(f"  {label:24s} {v:.3f}")
delta = np.mean(acc["multifoveal"]) - np.mean(acc["single"])
print(f"\nmulti-foveal - single = {delta:+.3f}  "
      f"({'multi-foveal covers more objects' if delta > 0 else 'no coverage gain'})")
