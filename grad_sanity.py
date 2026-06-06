"""SoftWhere (P3) — gradient-flow sanity check.

Proves the central mechanism claim: gradients from an extractor-feature loss
reach the TokenLearner selector head, despite LookWhere's selection being
non-differentiable (a hard top-k of integer indices feeding a frozen extractor).

The barrier in `LookWhereDownstream.forward`:
  1. the selector runs under `torch.no_grad()`,
  2. `torch.topk(...).indices` are integers (non-differentiable),
  3. the extractor gathers patches by index (gather passes no grad to selection).

The fix here (straight-through estimator):
  - run the selector WITH grad (backbone stays frozen via requires_grad_(False)),
  - build a differentiable soft gate from the aggregate importance map,
  - keep the hard top-k for the forward index set,
  - gate_st = gate_hard + (gate_soft - gate_soft.detach())  -> hard forward, soft backward,
  - multiply the gathered patch embeddings by gate_st (via Extractor's keep_gate).

Honest limitation (state in the pitch): the ST gate only RE-WEIGHTS selected
patches; excluded patches get no gradient (their features are never computed).
It can nudge the ranking via the soft aggregate over the full grid but cannot
directly "pull in" an excluded patch. A full solution (perturbed / soft top-k)
is future work. This script only demonstrates that gradients flow to the head.

Run:  .venv/bin/python grad_sanity.py
"""
import torch

from modeling import LookWhereDownstream

# ----------------------------- settings -----------------------------
checkpoint = "lookwhere_dinov2.pt"
high_res_img_size = 518
k_ratio = 0.10
tl_variant = "v10"           # v10 (sigmoid) or v11 (softmax)
num_tokens = 4
tau = 0.0                    # soft-gate threshold (on the importance map)
temp = 1.0                   # soft-gate temperature
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------

num_patches = (high_res_img_size // 14) ** 2
k = int(k_ratio * num_patches)

lw = LookWhereDownstream(
    pretrained_params_path=checkpoint,
    high_res_size=high_res_img_size,
    num_classes=0,
    k=k,
    is_cls=True,
    device=device,
    head_type="tokenlearner",
    num_tokens=num_tokens,
    tl_variant=tl_variant,
    tl_agg="max",
)
lw.train()

# Freeze everything except the TokenLearner selector head. Freezing (rather than
# no_grad) is what lets gradients still flow through the frozen modules to reach
# the head's parameters.
head_params = set(lw.selector.head.parameters())
for p in lw.parameters():
    p.requires_grad_(p in head_params)

n_train = sum(p.requires_grad for p in lw.parameters())
print(f"trainable param tensors: {n_train} (TokenLearner head only)")

x = torch.randn(2, 3, high_res_img_size, high_res_img_size, device=device)

# --- selector WITH grad (no torch.no_grad) ---
sel = lw.selector(x)
selector_map = sel["selector_map"]           # (B, num_patches), grad-enabled

# --- straight-through gate ---
gate_soft = torch.sigmoid((selector_map - tau) / temp)   # (B, num_patches)
idx = torch.topk(selector_map, k=k, sorted=True).indices  # (B, k), detached integers
gate_hard = torch.zeros_like(selector_map).scatter_(1, idx, 1.0)
gate_st = gate_hard + (gate_soft - gate_soft.detach())    # hard fwd, soft bwd

# gather the gate values for the kept patches -> (B, k, 1)
keep_gate = torch.gather(gate_st, 1, idx).unsqueeze(-1)

# --- extractor with the straight-through gate on kept patches ---
feats = lw.extractor(
    x=x,
    selector_prefix_tokens=sel["prefix_tokens"],
    keep_patch_indices=idx,
    return_only_cls=True,
    keep_gate=keep_gate,
)                                              # (B, dim)

loss = feats.pow(2).mean()
loss.backward()

# --- verdict ---
print(f"\nloss = {loss.item():.4f}")
print("\nper-parameter grad norms (TokenLearner head):")
ok = True
for name, p in lw.selector.head.named_parameters():
    g = p.grad
    norm = None if g is None else g.norm().item()
    finite = g is not None and torch.isfinite(g).all().item()
    nonzero = g is not None and bool((g != 0).any().item())
    flag = "OK" if (g is not None and finite and nonzero) else "FAIL"
    ok = ok and flag == "OK"
    print(f"  [{flag}] head.{name:30s} grad_norm={norm}")

# frozen modules must NOT have accumulated gradients
leaked = [n for n, p in lw.named_parameters()
          if p not in head_params and p.grad is not None and (p.grad != 0).any()]
print(f"\nfrozen params with nonzero grad (should be empty): {leaked}")

assert ok, "some TokenLearner head params have no/zero/non-finite gradient"
assert not leaked, "gradient leaked into a frozen module"
print("\nPASS: gradients flow through hard top-k to the TokenLearner head only.")
