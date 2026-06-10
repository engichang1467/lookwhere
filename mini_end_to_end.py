"""SoftWhere experiment 3: mini end-to-end selector-signal training.

This is a small, local proxy for the proposal's full pretraining objective. It
keeps the selector backbone and extractor frozen, trains only the TokenLearner
selector head, and lets extractor-feature losses flow through a straight-through
gate on the selected patches.

The default teacher is the pretrained LookWhere MLP selector/extractor. That is
not the final paper teacher, but it is enough to test whether extractor-signal
gradients improve or destabilize the SoftWhere selector before spending full
ImageNet-scale compute.

Example:
  .venv/bin/python mini_end_to_end.py --tl-sr-mode conv --steps 1000
"""
import argparse

import torch
import torch.nn.functional as F

from modeling import LookWhereDownstream
from softwhere_experiment_utils import (
    DEFAULT_HIGH_RES,
    collect_image_paths,
    diversity_loss_from_head,
    image_transform,
    load_batch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lookwhere_dinov2.pt")
    parser.add_argument("--variant", choices=["v10", "v11"], default="v10")
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--tl-agg", choices=["max", "mean", "logsumexp"], default="max")
    parser.add_argument("--tl-sr-mode", choices=["none", "conv"], default="conv")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-images", type=int, default=512)
    parser.add_argument("--image-glob", default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--k-ratio", type=float, default=0.10)
    parser.add_argument("--map-temp", type=float, default=1.0)
    parser.add_argument("--gate-temp", type=float, default=0.25)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-pat", type=float, default=0.0)
    parser.add_argument("--lambda-map", type=float, default=1.0)
    parser.add_argument("--lambda-div", type=float, default=0.1)
    parser.add_argument("--init-head", default=None,
                        help="optional distilled TokenLearner head checkpoint")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def make_softwhere(args, device, k, is_cls=True):
    return LookWhereDownstream(
        pretrained_params_path=args.checkpoint,
        high_res_size=DEFAULT_HIGH_RES,
        num_classes=0,
        k=k,
        is_cls=is_cls,
        device=device,
        head_type="tokenlearner",
        num_tokens=args.num_tokens,
        tl_variant=args.variant,
        tl_agg=args.tl_agg,
        tl_sr_mode=args.tl_sr_mode,
    )


def freeze_for_head_training(model):
    head_params = set(model.selector.head.parameters())
    for p in model.parameters():
        p.requires_grad_(p in head_params)
    return model.selector.head


def extractor_features(model, images, selector_dict, keep_indices, keep_gate=None,
                       return_only_cls=True):
    return model.extractor(
        x=images,
        selector_prefix_tokens=selector_dict["prefix_tokens"],
        keep_patch_indices=keep_indices,
        return_only_cls=return_only_cls,
        keep_gate=keep_gate,
    )


def st_keep_gate(selector_map, keep_indices, gate_temp):
    kth = torch.gather(selector_map.detach(), 1, keep_indices[:, -1:])
    gate_soft = torch.sigmoid((selector_map - kth) / gate_temp)
    gate_hard = torch.zeros_like(selector_map).scatter_(1, keep_indices, 1.0)
    gate_st = gate_hard + (gate_soft - gate_soft.detach())
    return torch.gather(gate_st, 1, keep_indices).unsqueeze(-1)


def cosine_loss(student, teacher):
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    return 1.0 - (student * teacher).sum(dim=-1).mean()


def train_step(args, teacher, student, images, k):
    with torch.no_grad():
        t_sel = teacher.selector(images)
        t_idx = torch.topk(t_sel["selector_map"], k=k, sorted=True).indices
        t_cls = extractor_features(teacher, images, t_sel, t_idx, return_only_cls=True)
        t_patch = None
        if args.lambda_pat > 0:
            t_patch = extractor_features(teacher, images, t_sel, t_idx, return_only_cls=False)
        t_map_dist = F.softmax(t_sel["selector_map"] / args.map_temp, dim=-1)

    s_sel = student.selector(images)
    s_idx = torch.topk(s_sel["selector_map"], k=k, sorted=True).indices
    keep_gate = st_keep_gate(s_sel["selector_map"], s_idx, args.gate_temp)

    s_cls = extractor_features(student, images, s_sel, s_idx, keep_gate, return_only_cls=True)
    losses = {}
    losses["cls"] = cosine_loss(s_cls, t_cls)
    losses["map"] = F.kl_div(
        F.log_softmax(s_sel["selector_map"] / args.map_temp, dim=-1),
        t_map_dist,
        reduction="batchmean",
    )
    if args.lambda_pat > 0:
        s_patch = extractor_features(student, images, s_sel, s_idx, keep_gate, return_only_cls=False)
        losses["pat"] = F.mse_loss(s_patch.float(), t_patch.float())
    else:
        losses["pat"] = s_cls.new_tensor(0.0)
    losses["div"] = diversity_loss_from_head(student.selector.head)

    total = (
        args.lambda_cls * losses["cls"]
        + args.lambda_pat * losses["pat"]
        + args.lambda_map * losses["map"]
        + args.lambda_div * losses["div"]
    )
    return total, losses


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = DEFAULT_HIGH_RES // 14
    k = int(args.k_ratio * grid * grid)

    paths = collect_image_paths(args.image_glob, args.n_images, include_ice_cream=args.image_glob is None)
    if not paths:
        raise SystemExit("no training images found")
    transform = image_transform(DEFAULT_HIGH_RES)

    teacher = LookWhereDownstream(args.checkpoint, DEFAULT_HIGH_RES, 0, k, True, device, head_type="mlp")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = make_softwhere(args, device, k, is_cls=True)
    if args.init_head:
        student.selector.head.load_state_dict(torch.load(args.init_head, map_location=device, weights_only=True))
        print(f"initialized head from {args.init_head}")
    head = freeze_for_head_training(student)
    student.eval()
    student.selector.train()

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"training SoftWhere head on {len(paths)} images, steps={args.steps}, k={k}, tl_sr_mode={args.tl_sr_mode}")

    for step in range(args.steps):
        idx = torch.randint(0, len(paths), (args.batch_size,))
        batch_paths = [paths[i] for i in idx.tolist()]
        images = load_batch(batch_paths, transform, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss, parts = train_step(args, teacher, student, images, k)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if step % 25 == 0 or step == args.steps - 1:
            msg = " ".join(f"{name}={value.detach().float().item():.4f}" for name, value in parts.items())
            print(f"step={step:05d} total={loss.detach().float().item():.4f} {msg}")

    out = args.out
    if out is None:
        sr = "sr" if args.tl_sr_mode == "conv" else "lowres"
        out = f"softwhere_head_{args.variant}_{sr}_mini_e2e.pt"
    torch.save(head.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
