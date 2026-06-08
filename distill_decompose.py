"""SoftWhere (P3) — distillation-decomposition mini-experiment.

Turns the randomly-initialized TokenLearner selector head into a MEANINGFUL
multi-foveal selector, cheaply (minutes on one A100, no labels), by distilling
the pretrained LookWhere saliency into it.

Setup:
  - freeze the selector backbone AND the extractor,
  - train ONLY the TokenLearnerSelectorHead,
  - teacher = the pretrained LookWhere `selector_map` (the original MLP head),
    computed once under no_grad,
  - student = the SoftWhere aggregate importance map,
  - loss = KL( softmax(teacher/T) || log_softmax(student/T) ) over spatial
    positions  (+ optional cross-map diversity to discourage the S maps from
    collapsing onto one blob).

This is a deliberate overfit on a small image set — generalization is NOT the
claim. The artifact: after distillation the S foveal maps tile the teacher's
salient region into distinct sub-regions. Re-run experiment_softwhere.py with
--distilled <saved head .pt> to render the before/after figure.

Run:  .venv/bin/python distill_decompose.py --variant v10 --steps 600
"""
import argparse
import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from modeling import LookWhereDownstream

# ----------------------------- settings -----------------------------
checkpoint = "lookwhere_dinov2.pt"
high_res_img_size = 518
k_ratio = 0.10
imagenette_val = "/home/michael/ProjectE2/OpenTokenLearner/data/imagenette2-320/val"
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="v10", choices=["v10", "v11"])
parser.add_argument("--num_tokens", type=int, default=4)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--temp", type=float, default=1.0, help="softmax temperature")
parser.add_argument("--diversity", type=float, default=0.0,
                    help="weight on the cross-map diversity regularizer")
parser.add_argument("--n_images", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--image_glob", default=None,
                    help="override image source (e.g. ADE20K train) for in-domain "
                         "distillation; default = ice_cream + imagenette val")
parser.add_argument("--tag", default="", help="extra suffix on the saved filename")
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

# --- small image set: ice_cream + imagenette val, or a custom --image_glob ---
if args.image_glob:
    paths = sorted(glob.glob(args.image_glob))[: args.n_images]
else:
    paths = ["ice_cream.jpg"]
    paths += sorted(glob.glob(os.path.join(imagenette_val, "*", "*.JPEG")))[: args.n_images - 1]
images = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths]).to(device)
print(f"distilling on {len(images)} images, variant={args.variant}, "
      f"S={args.num_tokens}, steps={args.steps}")

# --- teacher: pretrained MLP-head saliency, computed once ---
lw_mlp = LookWhereDownstream(checkpoint, high_res_size=high_res_img_size, num_classes=0,
                             k=k, is_cls=True, device=device, head_type="mlp")
lw_mlp.eval()
with torch.no_grad():
    teacher = torch.cat([
        lw_mlp.selector(images[i:i + args.batch_size])["selector_map"]
        for i in range(0, len(images), args.batch_size)
    ])  # (N, grid*grid)
t_dist = F.softmax(teacher / args.temp, dim=-1).detach()
del lw_mlp

# --- student: TokenLearner head, only head trainable ---
lw_tl = LookWhereDownstream(checkpoint, high_res_size=high_res_img_size, num_classes=0,
                            k=k, is_cls=True, device=device, head_type="tokenlearner",
                            num_tokens=args.num_tokens, tl_variant=args.variant, tl_agg="max")
head = lw_tl.selector.head
for p in lw_tl.parameters():
    p.requires_grad_(False)
for p in head.parameters():
    p.requires_grad_(True)
lw_tl.selector.train()

opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)


def diversity_loss(attn):
    """Encourage the S maps to cover distinct regions (low pairwise overlap)."""
    b, s, h, w = attn.shape
    a = attn.reshape(b, s, h * w)
    a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)   # per-map distribution
    gram = torch.bmm(a, a.transpose(1, 2))          # (B, S, S) pairwise overlap
    off = gram - torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
    return off.sum(dim=(1, 2)).mean() / max(s * (s - 1), 1)


for step in range(args.steps):
    idx = torch.randint(0, len(images), (args.batch_size,), device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        student = lw_tl.selector(images[idx])["selector_map"]   # (B, grid*grid)
        s_logdist = F.log_softmax(student / args.temp, dim=-1)
        loss = F.kl_div(s_logdist, t_dist[idx], reduction="batchmean")
        if args.diversity > 0:
            loss = loss + args.diversity * diversity_loss(head._last_attn)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    if step % 50 == 0 or step == args.steps - 1:
        print(f"  step {step:4d}  kl={loss.item():.5f}")

# filename tags variant + diversity weight so a sweep does not overwrite.
div_tag = f"_div{args.diversity:g}"
out = f"softwhere_head_{args.variant}{div_tag}{args.tag}.pt"
torch.save(head.state_dict(), out)
print(f"\nsaved distilled head -> {out}")
print(f"render with:  .venv/bin/python experiment_softwhere.py --distilled {out}")
