# SoftWhere Next Experiments

These scripts set up the three next-step diagnostics from the proposal plan.
Run them from `softwhere/lookwhere` with the shared venv:

```bash
cd /home/michael/ProjectE2/softwhere/lookwhere
../.venv/bin/python <script>.py ...
```

## 1. Resolution-Parity TokenLearner

Goal: test whether the ADE20K negative result was caused by the current
TokenLearner selector emitting only an `11x11` map before bilinear upsampling.
The new `tl_sr_mode=conv` path gives each foveal map a learned super-resolution
refiner before the final `37x37` selector map.

```bash
../.venv/bin/python resolution_parity.py \
  --tl-sr-mode conv \
  --variant v10 \
  --diversity 1 \
  --stage both \
  --eval-ade20k
```

Gate: the SoftWhere aggregate should recover LookWhere coverage much better
than the low-resolution head did. If it still trails random, do not spend full
pretraining compute yet.

## 2. Selection-Policy Ablation

Goal: keep the trained selector fixed and test whether the coverage failure is
mostly caused by the way foveal maps are converted into top-`k` patches.

```bash
../.venv/bin/python selection_policy_ablation.py \
  --distilled softwhere_head_v10_sr_div1.pt \
  --tl-sr-mode conv \
  --variant v10
```

Policies reported:

- `lookwhere_single`
- `softwhere_agg`
- `per_map_topk`
- `per_map_nms`
- `distance_penalty`
- `random`

Gate: at least one multi-foveal policy should beat random and ideally beat
LookWhere on the small multi-object ADE20K subset before the coverage claim is
treated as positive.

## 3. Mini End-To-End Selector Signal

Goal: test whether extractor-feature gradients improve or destabilize the
selector before running ImageNet-scale pretraining. This trains only the
TokenLearner selector head; selector backbone, extractor, and teacher are frozen.

```bash
../.venv/bin/python mini_end_to_end.py \
  --tl-sr-mode conv \
  --variant v10 \
  --init-head softwhere_head_v10_sr_div1.pt \
  --steps 1000 \
  --lambda-cls 1 \
  --lambda-map 1 \
  --lambda-div 0.1
```

Optional patch-feature loss:

```bash
../.venv/bin/python mini_end_to_end.py \
  --tl-sr-mode conv \
  --variant v10 \
  --init-head softwhere_head_v10_sr_div1.pt \
  --steps 1000 \
  --lambda-cls 1 \
  --lambda-pat 0.1 \
  --lambda-map 1 \
  --lambda-div 0.1
```

Gate: feature loss should improve without collapsing teacher agreement or map
diversity. If it does not, frame the result as evidence against the extractor
signal rather than moving directly to the full sweep.
