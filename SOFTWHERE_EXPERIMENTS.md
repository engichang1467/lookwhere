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

## 4. Robustness Sweep For The NMS Win

Goal: test whether the positive `per_map_nms` result is stable across NMS
distance and paper-relevant selection budgets.

```bash
../.venv/bin/python nms_robustness_sweep.py \
  --head div1,softwhere_head_v10_sr_div1.pt,4,v10,conv \
  --k-values 16 72 128 136 \
  --nms-dists 1 2 3 4 \
  --out-csv softwhere_nms_robustness_div1.csv
```

To sweep diversity weights, first produce matching SR heads:

```bash
../.venv/bin/python resolution_parity.py \
  --tl-sr-mode conv \
  --variant v10 \
  --diversity 0 \
  --stage distill

../.venv/bin/python resolution_parity.py \
  --tl-sr-mode conv \
  --variant v10 \
  --diversity 0.5 \
  --stage distill

../.venv/bin/python resolution_parity.py \
  --tl-sr-mode conv \
  --variant v10 \
  --diversity 2 \
  --stage distill
```

Then evaluate them together:

```bash
../.venv/bin/python nms_robustness_sweep.py \
  --head div0,softwhere_head_v10_sr_div0.pt,4,v10,conv \
  --head div0p5,softwhere_head_v10_sr_div0.5.pt,4,v10,conv \
  --head div1,softwhere_head_v10_sr_div1.pt,4,v10,conv \
  --head div2,softwhere_head_v10_sr_div2.pt,4,v10,conv \
  --k-values 16 72 128 136 \
  --nms-dists 1 2 3 4 \
  --out-csv softwhere_nms_robustness_diversity.csv
```

To sweep number of foveal maps `S`, produce heads with different
`--num-tokens` values and include the matching value in each `--head` spec:

```bash
../.venv/bin/python resolution_parity.py \
  --tl-sr-mode conv \
  --variant v10 \
  --num-tokens 2 \
  --diversity 1 \
  --stage distill \
  --tag S2

../.venv/bin/python resolution_parity.py \
  --tl-sr-mode conv \
  --variant v10 \
  --num-tokens 8 \
  --diversity 1 \
  --stage distill \
  --tag S8

../.venv/bin/python nms_robustness_sweep.py \
  --head S2,softwhere_head_v10_sr_div1_S2.pt,2,v10,conv \
  --head S4,softwhere_head_v10_sr_div1.pt,4,v10,conv \
  --head S8,softwhere_head_v10_sr_div1_S8.pt,8,v10,conv \
  --k-values 16 72 128 136 \
  --nms-dists 1 2 3 4 \
  --out-csv softwhere_nms_robustness_S.csv
```

Gate: the `per_map_nms` advantage should persist for at least `k=72` and
`k=128`, with one reasonable NMS distance and without relying on a single lucky
diversity setting.

## 5. Sanity Check The Current Main Head

After the diversity and `S` sweeps, the current main coverage configuration is:

```text
v10 / TokenLearner-SR conv / S=4 / diversity=0 / per_map_nms / nms_dist=2
```

Run teacher agreement, map overlap, and ADE20K coverage for that head:

```bash
../.venv/bin/python resolution_parity.py \
  --tl-sr-mode conv \
  --variant v10 \
  --diversity 0 \
  --stage eval \
  --distilled softwhere_head_v10_sr_div0.pt \
  --eval-ade20k
```

Render the SR foveal maps:

```bash
../.venv/bin/python experiment_softwhere.py \
  --distilled softwhere_head_v10_sr_div0.pt \
  --tl-sr-mode conv \
  --variant v10 \
  --num-tokens 4
```

Then run the focused NMS coverage check:

```bash
../.venv/bin/python nms_robustness_sweep.py \
  --head div0,softwhere_head_v10_sr_div0.pt,4,v10,conv \
  --k-values 16 72 128 136 \
  --nms-dists 2 \
  --out-csv softwhere_nms_robustness_main_div0.csv
```

Gate: `div0` should have non-degenerate foveal maps, reasonable teacher
agreement, and coverage above LookWhere at `k=72` and `k=128`.

## 6. Mini End-To-End From The Main Head

Goal: test whether extractor-feature gradients help or hurt the current best
coverage selector. Use the `div0` SR head as initialization and keep explicit
diversity off, since the sweeps show `diversity=0` is strongest for coverage.

Conservative cls-feature run:

```bash
../.venv/bin/python mini_end_to_end.py \
  --tl-sr-mode conv \
  --variant v10 \
  --num-tokens 4 \
  --init-head softwhere_head_v10_sr_div0.pt \
  --steps 1000 \
  --lambda-cls 1 \
  --lambda-pat 0 \
  --lambda-map 1 \
  --lambda-div 0 \
  --out softwhere_head_v10_sr_div0_mini_e2e_cls.pt
```

Evaluate against the original `div0` head:

```bash
../.venv/bin/python nms_robustness_sweep.py \
  --head div0,softwhere_head_v10_sr_div0.pt,4,v10,conv \
  --head div0_e2e_cls,softwhere_head_v10_sr_div0_mini_e2e_cls.pt,4,v10,conv \
  --k-values 16 72 128 136 \
  --nms-dists 2 \
  --out-csv softwhere_nms_robustness_div0_vs_e2e_cls.csv
```

Also check teacher agreement and aggregate/multifoveal coverage:

```bash
../.venv/bin/python resolution_parity.py \
  --tl-sr-mode conv \
  --variant v10 \
  --num-tokens 4 \
  --stage eval \
  --distilled softwhere_head_v10_sr_div0_mini_e2e_cls.pt \
  --eval-ade20k
```

Optional patch-feature variant:

```bash
../.venv/bin/python mini_end_to_end.py \
  --tl-sr-mode conv \
  --variant v10 \
  --num-tokens 4 \
  --init-head softwhere_head_v10_sr_div0.pt \
  --steps 1000 \
  --lambda-cls 1 \
  --lambda-pat 0.1 \
  --lambda-map 1 \
  --lambda-div 0 \
  --out softwhere_head_v10_sr_div0_mini_e2e_cls_pat.pt
```

Compare all three:

```bash
../.venv/bin/python nms_robustness_sweep.py \
  --head div0,softwhere_head_v10_sr_div0.pt,4,v10,conv \
  --head e2e_cls,softwhere_head_v10_sr_div0_mini_e2e_cls.pt,4,v10,conv \
  --head e2e_cls_pat,softwhere_head_v10_sr_div0_mini_e2e_cls_pat.pt,4,v10,conv \
  --k-values 16 72 128 136 \
  --nms-dists 2 \
  --out-csv softwhere_nms_robustness_e2e_compare.csv
```

Gate: mini end-to-end training is useful only if it preserves the existing
coverage win at `k=72` and `k=128` and ideally improves teacher agreement or
the coverage margins. If coverage falls, keep the distilled `div0` head as the
main coverage result and treat extractor-signal training as a separate risk.

## 7. Sanity Check The Mini E2E CLS Head

The mini E2E CLS run improved coverage. Now verify that the improved head has
reasonable map quality and did not win through a strange artifact.

Teacher agreement, map overlap, and ADE20K aggregate/multifoveal coverage:

```bash
../.venv/bin/python resolution_parity.py \
  --tl-sr-mode conv \
  --variant v10 \
  --num-tokens 4 \
  --stage eval \
  --distilled softwhere_head_v10_sr_div0_mini_e2e_cls.pt \
  --eval-ade20k
```

Render the SR foveal maps:

```bash
../.venv/bin/python experiment_softwhere.py \
  --distilled softwhere_head_v10_sr_div0_mini_e2e_cls.pt \
  --tl-sr-mode conv \
  --variant v10 \
  --num-tokens 4
```

Optional focused comparison, if you want a fresh spreadsheet with just the two
current heads:

```bash
../.venv/bin/python nms_robustness_sweep.py \
  --head div0,softwhere_head_v10_sr_div0.pt,4,v10,conv \
  --head e2e_cls,softwhere_head_v10_sr_div0_mini_e2e_cls.pt,4,v10,conv \
  --k-values 16 72 128 136 \
  --nms-dists 2 \
  --out-csv softwhere_nms_robustness_div0_vs_e2e_cls_sanity.csv
```

Gate: keep `softwhere_head_v10_sr_div0_mini_e2e_cls.pt` as the main head only
if teacher agreement and map overlap remain interpretable and the visualization
still shows meaningful SR foveal maps.
