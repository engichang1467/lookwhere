"""SoftWhere experiment 1: TokenLearner resolution parity.

This runner compares the original low-resolution TokenLearner selector against
`tl_sr_mode=conv`, which gives the foveal maps the same pre-interpolation
super-resolution opportunity as LookWhere's MLP selector head.

Example:
  .venv/bin/python resolution_parity.py --tl-sr-mode conv --stage both --eval-ade20k
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from modeling import LookWhereDownstream
from softwhere_experiment_utils import (
    DEFAULT_HIGH_RES,
    collect_image_paths,
    connected_components,
    default_ade_root,
    diversity_loss_from_head,
    foveal_maps_at_grid,
    image_transform,
    load_batch,
    mask_to_patch_labels,
    object_recall,
    select_per_map_topk,
    topk_mask,
    topk_set,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["distill", "eval", "both"], default="both")
    parser.add_argument("--checkpoint", default="lookwhere_dinov2.pt")
    parser.add_argument("--variant", choices=["v10", "v11"], default="v10")
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--tl-agg", choices=["max", "mean", "logsumexp"], default="max")
    parser.add_argument("--tl-sr-mode", choices=["none", "conv"], default="conv")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--diversity", type=float, default=1.0)
    parser.add_argument("--n-images", type=int, default=64)
    parser.add_argument("--eval-images", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-glob", default=None)
    parser.add_argument("--distilled", default=None,
                        help="existing head checkpoint for --stage eval")
    parser.add_argument("--tag", default="")
    parser.add_argument("--eval-ade20k", action="store_true")
    parser.add_argument("--ade-root", default=str(default_ade_root()))
    parser.add_argument("--ade-images", type=int, default=500)
    parser.add_argument("--min-obj-patches", type=int, default=3)
    parser.add_argument("--max-obj-patches", type=int, default=25)
    parser.add_argument("--min-objects", type=int, default=3)
    return parser.parse_args()


def make_model(args, device, k):
    return LookWhereDownstream(
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


def output_name(args):
    sr = "sr" if args.tl_sr_mode == "conv" else "lowres"
    tag = f"_{args.tag}" if args.tag else ""
    return f"softwhere_head_{args.variant}_{sr}_div{args.diversity:g}{tag}.pt"


def distill(args, device, grid, k):
    transform = image_transform(DEFAULT_HIGH_RES)
    paths = collect_image_paths(args.image_glob, args.n_images, include_ice_cream=args.image_glob is None)
    images = load_batch(paths, transform, device)
    print(f"distilling {args.variant}/{args.tl_sr_mode} on {len(paths)} images, k={k}")

    teacher = LookWhereDownstream(args.checkpoint, DEFAULT_HIGH_RES, 0, k, True, device, head_type="mlp")
    teacher.eval()
    with torch.no_grad():
        t_maps = []
        for i in range(0, len(images), args.batch_size):
            t_maps.append(teacher.selector(images[i:i + args.batch_size])["selector_map"])
        t_maps = torch.cat(t_maps)
    t_dist = F.softmax(t_maps / args.temp, dim=-1).detach()
    del teacher

    student = make_model(args, device, k)
    head = student.selector.head
    for p in student.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(True)
    student.selector.train()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    for step in range(args.steps):
        idx = torch.randint(0, len(images), (args.batch_size,), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            s_map = student.selector(images[idx])["selector_map"]
            loss = F.kl_div(
                F.log_softmax(s_map / args.temp, dim=-1),
                t_dist[idx],
                reduction="batchmean",
            )
            if args.diversity > 0:
                loss = loss + args.diversity * diversity_loss_from_head(head)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss={loss.item():.5f}")

    out = output_name(args)
    torch.save(head.state_dict(), out)
    print(f"saved {out}")
    return out


def evaluate_teacher_agreement(args, device, k, distilled):
    transform = image_transform(DEFAULT_HIGH_RES)
    paths = collect_image_paths(args.image_glob, args.eval_images, include_ice_cream=False)
    print(f"teacher-agreement eval on {len(paths)} images")

    teacher = LookWhereDownstream(args.checkpoint, DEFAULT_HIGH_RES, 0, k, True, device, head_type="mlp")
    student = make_model(args, device, k)
    student.selector.head.load_state_dict(torch.load(distilled, map_location=device, weights_only=True))
    teacher.eval()
    student.eval()

    recalls, ious, rand_recalls, overlaps = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            batch = load_batch(paths[i:i + args.batch_size], transform, device)
            t_map = teacher.selector(batch)["selector_map"]
            s_map = student.selector(batch)["selector_map"]
            t_mask = topk_mask(t_map, k)
            s_mask = topk_mask(s_map, k)
            r_mask = topk_mask(torch.rand_like(s_map), k)
            inter = (s_mask & t_mask).sum(-1).float()
            union = (s_mask | t_mask).sum(-1).float()
            recalls.append((inter / k).cpu())
            ious.append((inter / union).cpu())
            rand_recalls.append(((r_mask & t_mask).sum(-1).float() / k).cpu())
            overlaps.append(float(diversity_loss_from_head(student.selector.head).detach().cpu()))

    print("\n--- teacher agreement ---")
    print(f"SoftWhere recall: {torch.cat(recalls).mean().item():.3f}")
    print(f"SoftWhere IoU:    {torch.cat(ious).mean().item():.3f}")
    print(f"random recall:    {torch.cat(rand_recalls).mean().item():.3f}")
    print(f"map overlap:      {np.mean(overlaps):.3f}  (lower is more diverse)")


def evaluate_ade20k(args, device, grid, k, distilled):
    ade_root = Path(args.ade_root)
    img_paths = sorted(glob.glob(str(ade_root / "images/validation/*.jpg")))[:args.ade_images]
    if not img_paths:
        raise SystemExit(f"no ADE20K validation images found under {ade_root}")
    transform = image_transform(DEFAULT_HIGH_RES)
    teacher = LookWhereDownstream(args.checkpoint, DEFAULT_HIGH_RES, 0, k, True, device, head_type="mlp")
    student = make_model(args, device, k)
    student.selector.head.load_state_dict(torch.load(distilled, map_location=device, weights_only=True))
    teacher.eval()
    student.eval()

    acc = {"lookwhere": [], "softwhere_agg": [], "softwhere_multifoveal": [], "random": []}
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
            single = teacher.selector(x)["selector_map"][0]
            agg = student.selector(x)["selector_map"][0]
            maps_s = foveal_maps_at_grid(student.selector.head, grid)[0]
            random_scores = torch.rand(grid * grid, generator=rng, device=device)

            acc["lookwhere"].append(object_recall(comps, topk_set(single, k)))
            acc["softwhere_agg"].append(object_recall(comps, topk_set(agg, k)))
            acc["softwhere_multifoveal"].append(object_recall(comps, select_per_map_topk(maps_s, k)))
            acc["random"].append(object_recall(comps, topk_set(random_scores, k)))

    print(f"\nADE20K multi-object images: {n_multi}/{len(img_paths)}")
    print("--- object-level coverage recall ---")
    for key, values in acc.items():
        print(f"{key:24s} {np.mean(values):.3f}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = DEFAULT_HIGH_RES // 14
    k = int(0.10 * grid * grid)
    distilled = args.distilled

    if args.stage in ("distill", "both"):
        distilled = distill(args, device, grid, k)
    if args.stage in ("eval", "both"):
        if not distilled:
            raise SystemExit("--distilled is required for --stage eval")
        evaluate_teacher_agreement(args, device, k, distilled)
        if args.eval_ade20k:
            evaluate_ade20k(args, device, grid, k, distilled)


if __name__ == "__main__":
    main()
