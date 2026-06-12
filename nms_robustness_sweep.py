"""SoftWhere next step: robustness sweep for per-map NMS coverage.

The first two diagnostics showed that TokenLearner-SR plus per-map NMS can beat
the LookWhere single-map selector on ADE20K small-object coverage. This script
tests whether that win is stable across NMS distance, selection budget k, and
multiple distilled heads.

Head spec format:
  label,path,num_tokens,variant,tl_sr_mode

Example:
  .venv/bin/python nms_robustness_sweep.py \
    --head div1,softwhere_head_v10_sr_div1.pt,4,v10,conv \
    --k-values 16 72 128 136 \
    --nms-dists 1 2 3 4
"""
import argparse
import csv
import glob
from dataclasses import dataclass
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
    select_per_map_nms,
    select_per_map_topk,
    topk_set,
)


@dataclass(frozen=True)
class HeadSpec:
    label: str
    path: str
    num_tokens: int
    variant: str
    tl_sr_mode: str


def parse_head(spec):
    parts = spec.split(",")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "head spec must be label,path,num_tokens,variant,tl_sr_mode")
    label, path, num_tokens, variant, tl_sr_mode = parts
    if variant not in {"v10", "v11"}:
        raise argparse.ArgumentTypeError("variant must be v10 or v11")
    if tl_sr_mode not in {"none", "conv"}:
        raise argparse.ArgumentTypeError("tl_sr_mode must be none or conv")
    return HeadSpec(label, path, int(num_tokens), variant, tl_sr_mode)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lookwhere_dinov2.pt")
    parser.add_argument("--head", action="append", type=parse_head, required=True,
                        help="repeatable: label,path,num_tokens,variant,tl_sr_mode")
    parser.add_argument("--tl-agg", choices=["max", "mean", "logsumexp"], default="max")
    parser.add_argument("--ade-root", default=str(default_ade_root()))
    parser.add_argument("--n-images", type=int, default=500)
    parser.add_argument("--k-values", nargs="+", type=int, default=[16, 72, 128, 136])
    parser.add_argument("--k-ratios", nargs="*", type=float, default=[],
                        help="optional extra k values as ratios of the 37x37 grid")
    parser.add_argument("--nms-dists", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--min-obj-patches", type=int, default=3)
    parser.add_argument("--max-obj-patches", type=int, default=25)
    parser.add_argument("--min-objects", type=int, default=3)
    parser.add_argument("--out-csv", default="softwhere_nms_robustness.csv")
    return parser.parse_args()


def make_softwhere(args, spec, device, k):
    model = LookWhereDownstream(
        pretrained_params_path=args.checkpoint,
        high_res_size=DEFAULT_HIGH_RES,
        num_classes=0,
        k=k,
        is_cls=True,
        device=device,
        head_type="tokenlearner",
        num_tokens=spec.num_tokens,
        tl_variant=spec.variant,
        tl_agg=args.tl_agg,
        tl_sr_mode=spec.tl_sr_mode,
    )
    model.selector.head.load_state_dict(torch.load(spec.path, map_location=device, weights_only=True))
    model.eval()
    return model


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = DEFAULT_HIGH_RES // 14
    num_patches = grid * grid
    extra_ks = [int(r * num_patches) for r in args.k_ratios]
    k_values = sorted({k for k in args.k_values + extra_ks if 0 < k <= num_patches})

    ade_root = Path(args.ade_root)
    img_paths = sorted(glob.glob(str(ade_root / "images/validation/*.jpg")))[:args.n_images]
    if not img_paths:
        raise SystemExit(f"no ADE20K validation images found under {ade_root}")

    transform = image_transform(DEFAULT_HIGH_RES)
    lookwhere = LookWhereDownstream(args.checkpoint, DEFAULT_HIGH_RES, 0, max(k_values),
                                    True, device, head_type="mlp")
    lookwhere.eval()
    softwhere_models = {
        spec.label: (spec, make_softwhere(args, spec, device, max(k_values)))
        for spec in args.head
    }

    rows = []
    rng = torch.Generator(device=device).manual_seed(0)
    print(f"evaluating {len(img_paths)} ADE20K images, k_values={k_values}, nms={args.nms_dists}")

    with torch.no_grad():
        for k in k_values:
            accum = {}
            n_multi = 0
            for path in img_paths:
                mask_path = ade_root / "annotations/validation" / Path(path).with_suffix(".png").name
                labels = mask_to_patch_labels(Image.open(mask_path), DEFAULT_HIGH_RES)
                comps = connected_components(labels, args.min_obj_patches, args.max_obj_patches)
                if len(comps) < args.min_objects:
                    continue
                n_multi += 1

                x = load_batch([path], transform, device)
                lw_scores = lookwhere.selector(x)["selector_map"][0]
                random_scores = torch.rand(num_patches, generator=rng, device=device)
                lookwhere_rec = object_recall(comps, topk_set(lw_scores, k))
                random_rec = object_recall(comps, topk_set(random_scores, k))

                for label, (spec, model) in softwhere_models.items():
                    sw_scores = model.selector(x)["selector_map"][0]
                    maps_s = foveal_maps_at_grid(model.selector.head, grid)[0]
                    base = accum.setdefault(label, {
                        "lookwhere": [],
                        "random": [],
                        "agg": [],
                        "per_map_topk": [],
                        "nms": {d: [] for d in args.nms_dists},
                    })
                    base["lookwhere"].append(lookwhere_rec)
                    base["random"].append(random_rec)
                    base["agg"].append(object_recall(comps, topk_set(sw_scores, k)))
                    base["per_map_topk"].append(object_recall(comps, select_per_map_topk(maps_s, k)))
                    for dist in args.nms_dists:
                        selected = select_per_map_nms(maps_s, k, grid, min_dist=dist)
                        base["nms"][dist].append(object_recall(comps, selected))

            for label, values in accum.items():
                spec = softwhere_models[label][0]
                for dist in args.nms_dists:
                    row = {
                        "head": label,
                        "path": spec.path,
                        "variant": spec.variant,
                        "num_tokens": spec.num_tokens,
                        "tl_sr_mode": spec.tl_sr_mode,
                        "k": k,
                        "k_ratio": k / num_patches,
                        "nms_dist": dist,
                        "n_multi": n_multi,
                        "lookwhere_single": mean_or_nan(values["lookwhere"]),
                        "softwhere_agg": mean_or_nan(values["agg"]),
                        "per_map_topk": mean_or_nan(values["per_map_topk"]),
                        "per_map_nms": mean_or_nan(values["nms"][dist]),
                        "random": mean_or_nan(values["random"]),
                    }
                    rows.append(row)
                    print(
                        f"{label} k={k:3d} nms={dist}: "
                        f"nms={row['per_map_nms']:.4f} "
                        f"lw={row['lookwhere_single']:.4f} "
                        f"rand={row['random']:.4f}")

    fieldnames = [
        "head", "path", "variant", "num_tokens", "tl_sr_mode", "k", "k_ratio",
        "nms_dist", "n_multi", "lookwhere_single", "softwhere_agg",
        "per_map_topk", "per_map_nms", "random",
    ]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
