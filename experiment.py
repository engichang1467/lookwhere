"""
Playground for the pretrained LookWhere checkpoints (lookwhere_dinov2.pt / lookwhere_franca.pt).

Run:  .venv/bin/python experiment.py
Then tweak the user settings below and re-run.

What it does:
  1. Loads a LookWhere model from a .pt checkpoint.
  2. Runs the SELECTOR to get a "where to look" saliency map, and saves an
     overlay PNG (selector_overlay.png) so you can see which patches it picks.
  3. Runs the full model to extract a feature / logits, and prints the shapes.
"""
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from modeling import LookWhereDownstream

# ----------------------------- user settings -----------------------------
checkpoint = "lookwhere_dinov2.pt"   # or "lookwhere_franca.pt"
image_path = "ice_cream.jpg"
high_res_img_size = 518              # must be divisible by 14
k_ratio = 0.10                       # fraction of patches the extractor sees (0-1)
num_classes = 0                      # 0 = return features; >0 attaches a linear head
is_classification = True             # True: CLS/img-level; False: per-patch/segmentation
device = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------------------------------

assert high_res_img_size % 14 == 0
num_high_res_patches = (high_res_img_size // 14) ** 2
k = int(k_ratio * num_high_res_patches)
grid = high_res_img_size // 14

transform = transforms.Compose([
    transforms.Resize(high_res_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(high_res_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                         std=torch.tensor([0.229, 0.224, 0.225])),
])
pil = Image.open(image_path).convert("RGB")
image = transform(pil).unsqueeze(0).to(device)
print(f"input {tuple(image.shape)} | k={k}/{num_high_res_patches} | grid={grid}x{grid}")

lw = LookWhereDownstream(
    pretrained_params_path=checkpoint,
    high_res_size=high_res_img_size,
    num_classes=num_classes,
    k=k,
    is_cls=is_classification,
    device=device,
)
lw.eval()

with torch.no_grad():
    out = lw(image)
    sel = lw.selector(image)["selector_map"]  # (1, num_high_res_patches)

print("model output shape:", tuple(out.shape))
print("selector_map shape:", tuple(sel.shape))

# --- visualize: where does the selector look? ---
sel_grid = sel.reshape(grid, grid).float().cpu()
# the top-k patches that the extractor actually receives
topk_idx = torch.topk(sel.reshape(-1), k=k).indices.cpu().numpy()
mask = np.zeros(grid * grid, dtype=np.float32)
mask[topk_idx] = 1.0
mask = mask.reshape(grid, grid)

# normalize saliency to 0..1 for a heatmap
s = sel_grid.numpy()
s = (s - s.min()) / (s.max() - s.min() + 1e-8)

# upsample maps to image size
def up(arr):
    return np.array(Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (high_res_img_size, high_res_img_size), Image.NEAREST))

base = np.array(pil.resize((high_res_img_size, high_res_img_size))).astype(np.float32)
heat = up(s).astype(np.float32)
# red heatmap overlay
overlay = base.copy()
overlay[..., 0] = np.clip(0.5 * base[..., 0] + 0.5 * heat, 0, 255)
overlay[..., 1] = 0.5 * base[..., 1]
overlay[..., 2] = 0.5 * base[..., 2]
# dim patches that are NOT selected (top-k)
keep = up(mask)[..., None] / 255.0
overlay = overlay * (0.35 + 0.65 * keep)
Image.fromarray(overlay.astype(np.uint8)).save("selector_overlay.png")
print("saved selector_overlay.png  (bright = selected top-k patches, red = high saliency)")
