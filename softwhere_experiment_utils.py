"""Shared helpers for the SoftWhere diagnostic experiments.

The scripts in this folder are intentionally small entry points. This module
keeps their data paths, map extraction, coverage metric, and selection policies
consistent.
"""
import glob
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision import transforms


PATCH_SIZE = 14
DEFAULT_HIGH_RES = 518


def project_root():
    return Path(__file__).resolve().parents[1]


def default_imagenette_val():
    otl = os.environ.get("OTL_PATH")
    if otl:
        return Path(otl) / "data/imagenette2-320/val"
    return project_root() / "Open-TokenLearner/data/imagenette2-320/val"


def default_ade_root():
    return Path(__file__).resolve().parent / "ade_data/ADEChallengeData2016"


def image_transform(high_res_img_size=DEFAULT_HIGH_RES):
    return transforms.Compose([
        transforms.Resize(high_res_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(high_res_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                             std=torch.tensor([0.229, 0.224, 0.225])),
    ])


def collect_image_paths(image_glob=None, n_images=128, include_ice_cream=False):
    if image_glob:
        paths = sorted(glob.glob(image_glob))
    else:
        paths = sorted(glob.glob(str(default_imagenette_val() / "*" / "*.JPEG")))
    if include_ice_cream:
        ice = Path(__file__).resolve().parent / "ice_cream.jpg"
        paths = [str(ice)] + paths
    return paths[:n_images]


def load_batch(paths, transform, device):
    images = [transform(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(images).to(device)


def topk_mask(scores, k):
    idx = torch.topk(scores, k=k, dim=-1).indices
    return torch.zeros_like(scores).scatter_(1, idx, 1.0).bool()


def topk_set(scores_1d, k):
    return set(torch.topk(scores_1d, k=k).indices.detach().cpu().tolist())


def foveal_maps_at_grid(selector_head, grid):
    """Return cached per-fovea maps at a target patch grid.

    Uses the super-resolved maps when the TokenLearner-SR head is active;
    otherwise falls back to the low-resolution TokenLearner maps.
    """
    attn = getattr(selector_head, "_last_attn_sr", None)
    if attn is None:
        attn = selector_head._last_attn
    if attn is None:
        raise RuntimeError("selector head has no cached foveal maps; run selector first")
    maps = F.interpolate(attn.float(), size=(grid, grid), mode="bilinear", align_corners=False)
    return maps.reshape(maps.shape[0], maps.shape[1], grid * grid)


def pairwise_overlap(attn):
    """Mean off-diagonal histogram intersection for positive S maps."""
    b, s, h, w = attn.shape
    p = attn.reshape(b, s, h * w)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    inter = torch.minimum(p.unsqueeze(2), p.unsqueeze(1)).sum(-1)
    eye = torch.eye(s, device=attn.device).bool()
    return inter[:, ~eye].mean()


def diversity_loss_from_head(selector_head):
    attn = getattr(selector_head, "_last_attn_sr", None)
    if attn is None:
        attn = selector_head._last_attn
    return pairwise_overlap(attn)


def mask_to_patch_labels(mask_pil, high_res_img_size=DEFAULT_HIGH_RES):
    grid = high_res_img_size // PATCH_SIZE
    m = transforms.functional.resize(mask_pil, high_res_img_size,
                                     interpolation=transforms.InterpolationMode.NEAREST)
    m = transforms.functional.center_crop(m, high_res_img_size)
    arr = np.array(m)
    arr = arr.reshape(grid, PATCH_SIZE, grid, PATCH_SIZE)
    out = np.zeros((grid, grid), dtype=np.int64)
    for i in range(grid):
        for j in range(grid):
            vals, cnts = np.unique(arr[i, :, j, :], return_counts=True)
            out[i, j] = vals[cnts.argmax()]
    return out


def connected_components(labels, min_obj_patches=3, max_obj_patches=25):
    comps = []
    for cls in np.unique(labels):
        if cls == 0:
            continue
        lab, n = ndimage.label(labels == cls)
        for c in range(1, n + 1):
            idxs = np.flatnonzero(lab.ravel() == c)
            if min_obj_patches <= len(idxs) <= max_obj_patches:
                comps.append(set(idxs.tolist()))
    return comps


def object_recall(comps, selected):
    if not comps:
        return None
    return sum(1 for c in comps if c & selected) / len(comps)


def select_per_map_topk(maps_s, k):
    s = maps_s.shape[0]
    per = max(1, k // s)
    selected = set()
    for i in range(s):
        selected |= topk_set(maps_s[i], per)
    if len(selected) < k:
        for idx in torch.topk(maps_s.amax(0), k=maps_s.shape[-1]).indices.detach().cpu().tolist():
            selected.add(idx)
            if len(selected) >= k:
                break
    return selected


def select_per_map_nms(maps_s, k, grid, min_dist=2):
    s = maps_s.shape[0]
    per = max(1, math.ceil(k / s))
    selected = []
    selected_set = set()

    def far_enough(idx):
        y, x = divmod(idx, grid)
        for kept in selected:
            ky, kx = divmod(kept, grid)
            if max(abs(y - ky), abs(x - kx)) < min_dist:
                return False
        return True

    for i in range(s):
        for idx in torch.argsort(maps_s[i], descending=True).detach().cpu().tolist():
            if idx not in selected_set and far_enough(idx):
                selected.append(idx)
                selected_set.add(idx)
            if len(selected_set) >= min(k, (i + 1) * per):
                break
    if len(selected_set) < k:
        for idx in torch.argsort(maps_s.amax(0), descending=True).detach().cpu().tolist():
            if idx not in selected_set:
                selected_set.add(idx)
            if len(selected_set) >= k:
                break
    return selected_set


def select_distance_penalty(scores, k, grid, penalty=0.25, radius=3):
    scores = scores.detach().float().cpu().clone()
    selected = []
    selected_set = set()
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).float()
    work = scores.clone()
    for _ in range(k):
        idx = int(torch.argmax(work).item())
        selected.append(idx)
        selected_set.add(idx)
        dist = torch.cdist(coords[idx:idx + 1], coords).squeeze(0)
        work = work - penalty * torch.exp(-(dist ** 2) / (2 * radius ** 2))
        work[idx] = -torch.inf
    return selected_set
