"""SoftWhere (P3) — teacher-agreement proxy coverage metric.

The proposal's coverage analysis wants "recall of GT object pixels inside
selected patches." No segmentation masks are available locally (imagenette /
tiny-imagenet are classification-only), so the REAL metric is not computable
here — it is specified as a post-pitch next step at the bottom of this file.

What we CAN compute now, with no masks, is a proxy that supports the
"SoftWhere matches LookWhere's *where*" claim:

  - pseudo-foreground = the pretrained LookWhere teacher's top-k patches,
  - report recall = |SoftWhere_topk ∩ Teacher_topk| / |Teacher_topk|,
    and IoU = |∩| / |∪|, averaged over images, at k_ratio = 0.10.

Label this honestly as "agreement with teacher saliency," NOT object coverage.
For reference we also report the agreement of a random selector (lower bound)
and the expected chance level (k / num_patches).

Run:  .venv/bin/python coverage_proxy.py --distilled softwhere_head_v10.pt --variant v10
"""
import argparse
import glob
import os

import torch
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
parser.add_argument("--distilled", default=None,
                    help="distilled TokenLearner head .pt (else random head)")
parser.add_argument("--n_images", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

grid = high_res_img_size // 14
num_patches = grid * grid
k = int(k_ratio * num_patches)

transform = transforms.Compose([
    transforms.Resize(high_res_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(high_res_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                         std=torch.tensor([0.229, 0.224, 0.225])),
])
paths = sorted(glob.glob(os.path.join(imagenette_val, "*", "*.JPEG")))[: args.n_images]
print(f"evaluating teacher-agreement on {len(paths)} images, k={k}/{num_patches}")


def topk_mask(selector_map):
    idx = torch.topk(selector_map, k=k, dim=-1).indices
    return torch.zeros_like(selector_map).scatter_(1, idx, 1.0).bool()


lw_mlp = LookWhereDownstream(checkpoint, high_res_size=high_res_img_size, num_classes=0,
                             k=k, is_cls=True, device=device, head_type="mlp")
lw_mlp.eval()
lw_tl = LookWhereDownstream(checkpoint, high_res_size=high_res_img_size, num_classes=0,
                            k=k, is_cls=True, device=device, head_type="tokenlearner",
                            num_tokens=args.num_tokens, tl_variant=args.variant, tl_agg="max")
if args.distilled:
    lw_tl.selector.head.load_state_dict(
        torch.load(args.distilled, map_location=device, weights_only=True))
    print(f"loaded distilled head: {args.distilled}")
else:
    print("using RANDOM (untrained) head")
lw_tl.eval()

recalls, ious, rand_recalls = [], [], []
with torch.no_grad():
    for i in range(0, len(paths), args.batch_size):
        batch = torch.stack([transform(Image.open(p).convert("RGB"))
                             for p in paths[i:i + args.batch_size]]).to(device)
        t_mask = topk_mask(lw_mlp.selector(batch)["selector_map"])
        s_mask = topk_mask(lw_tl.selector(batch)["selector_map"])
        r_mask = topk_mask(torch.rand(batch.shape[0], num_patches, device=device))

        inter = (s_mask & t_mask).sum(-1).float()
        union = (s_mask | t_mask).sum(-1).float()
        recalls.append((inter / t_mask.sum(-1).float()).cpu())
        ious.append((inter / union).cpu())
        rand_inter = (r_mask & t_mask).sum(-1).float()
        rand_recalls.append((rand_inter / t_mask.sum(-1).float()).cpu())

recall = torch.cat(recalls).mean().item()
iou = torch.cat(ious).mean().item()
rand_recall = torch.cat(rand_recalls).mean().item()

print("\n--- teacher-agreement (proxy for coverage) ---")
print(f"SoftWhere vs LookWhere teacher:  recall={recall:.3f}  IoU={iou:.3f}")
print(f"random selector vs teacher:      recall={rand_recall:.3f}  (lower bound)")
print(f"chance level (k/num_patches):    {k / num_patches:.3f}")
print("\nNOTE: this is agreement with teacher saliency, NOT object coverage.")

print("""
=== Real coverage metric (post-pitch next step) ===
Dataset: ADE20K val (2000 imgs, small) or COCO-Stuff val — both have masks.
Procedure:
  1. resize image to high_res=518; rasterize the GT mask to the 37x37 patch grid
     (a patch is foreground if any GT object pixel falls in its 14x14 region).
  2. selected = top-k patches at k_ratio=0.10.
  3. coverage recall = |foreground ∩ selected| / |foreground|, averaged.
  4. compare LookWhere single-map vs SoftWhere multi-foveal (the proposal's
     hypothesis: multi-foveal improves recall on multi-object images).
Not run here: neither dataset is present locally; download + mask rasterization
is ~half a day and is deferred to after the pitch.
""")
