"""SoftWhere experiment 2: selection-policy ablation on ADE20K coverage.

This script keeps the selector weights fixed and changes only how the foveal
maps are converted into k selected high-resolution patches.

Example:
  .venv/bin/python selection_policy_ablation.py \
    --distilled softwhere_head_v10_sr_div1.pt --tl-sr-mode conv
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from modeling import LookWhereDownstream
from softwhere_experiment_utils import (
    DEFAULT_HIGH_RES,
    connected_components,
    default_ade_root,
    foveal_maps_at_grid,
    image_transform,
    load_batch,
    mask_to_patch_labels,
    object_recall,
    select_distance_penalty,
    select_per_map_nms,
    select_per_map_topk,
    topk_set,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lookwhere_dinov2.pt")
    parser.add_argument("--distilled", required=True)
    parser.add_argument("--variant", choices=["v10", "v11"], default="v10")
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--tl-agg", choices=["max", "mean", "logsumexp"], default="max")
    parser.add_argument("--tl-sr-mode", choices=["none", "conv"], default="none")
    parser.add_argument("--ade-root", default=str(default_ade_root()))
    parser.add_argument("--n-images", type=int, default=500)
    parser.add_argument("--k-ratio", type=float, default=0.10)
    parser.add_argument("--min-obj-patches", type=int, default=3)
    parser.add_argument("--max-obj-patches", type=int, default=25)
    parser.add_argument("--min-objects", type=int, default=3)
    parser.add_argument("--nms-dist", type=int, default=2)
    parser.add_argument("--distance-penalty", type=float, default=0.25)
    parser.add_argument("--distance-radius", type=int, default=3)
    parser.add_argument("--policies", nargs="+", default=[
        "lookwhere_single",
        "softwhere_agg",
        "per_map_topk",
        "per_map_nms",
        "distance_penalty",
        "random",
    ])
    return parser.parse_args()


def make_student(args, device, k):
    model = LookWhereDownstream(
        pretrained_params_path=args.checkpoint,
        high_res_size=DEFAULT_HIGH_RES,
        num_classes=0,
        k=k,
        is_cls=True,
        device=device,
        head_type="tokenlearner",
        num_tokens=args.num_tokens,
        tl_variant=args.variant,
        tl_agg=args.tl_agg,
        tl_sr_mode=args.tl_sr_mode,
    )
    model.selector.head.load_state_dict(torch.load(args.distilled, map_location=device, weights_only=True))
    model.eval()
    return model


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = DEFAULT_HIGH_RES // 14
    num_patches = grid * grid
    k = int(args.k_ratio * num_patches)

    ade_root = Path(args.ade_root)
    img_paths = sorted(glob.glob(str(ade_root / "images/validation/*.jpg")))[:args.n_images]
    if not img_paths:
        raise SystemExit(f"no ADE20K validation images found under {ade_root}")

    transform = image_transform(DEFAULT_HIGH_RES)
    lookwhere = LookWhereDownstream(args.checkpoint, DEFAULT_HIGH_RES, 0, k, True, device, head_type="mlp")
    softwhere = make_student(args, device, k)
    lookwhere.eval()

    scores = {policy: [] for policy in args.policies}
    n_multi = 0
    rng = torch.Generator(device=device).manual_seed(0)

    with torch.no_grad():
        for path in img_paths:
            mask_path = ade_root / "annotations/validation" / Path(path).with_suffix(".png").name
            labels = mask_to_patch_labels(Image.open(mask_path), DEFAULT_HIGH_RES)
            comps = connected_components(labels, args.min_obj_patches, args.max_obj_patches)
            if len(comps) < args.min_objects:
                continue
            n_multi += 1

            x = load_batch([path], transform, device)
            lw_scores = lookwhere.selector(x)["selector_map"][0]
            sw_scores = softwhere.selector(x)["selector_map"][0]
            maps_s = foveal_maps_at_grid(softwhere.selector.head, grid)[0]
            random_scores = torch.rand(num_patches, generator=rng, device=device)

            selections = {
                "lookwhere_single": topk_set(lw_scores, k),
                "softwhere_agg": topk_set(sw_scores, k),
                "per_map_topk": select_per_map_topk(maps_s, k),
                "per_map_nms": select_per_map_nms(maps_s, k, grid, args.nms_dist),
                "distance_penalty": select_distance_penalty(
                    sw_scores, k, grid, args.distance_penalty, args.distance_radius),
                "random": topk_set(random_scores, k),
            }
            for policy in args.policies:
                scores[policy].append(object_recall(comps, selections[policy]))

    print(f"ADE20K multi-object images evaluated: {n_multi}/{len(img_paths)}")
    print(f"k={k}/{num_patches}, S={args.num_tokens}, tl_sr_mode={args.tl_sr_mode}")
    print("\npolicy,object_recall")
    for policy in args.policies:
        vals = scores[policy]
        print(f"{policy},{np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
