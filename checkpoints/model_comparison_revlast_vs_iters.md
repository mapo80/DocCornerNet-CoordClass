# Model leaderboard (rev-last vs rev-new vs val_clean_iter4_mix)

**How the leaderboard is ranked**
- Primary metric: **Mean IoU** (higher is better)
- Tie‑breakers: **R@95**, then **R@99**, then **lower err p95**
- `det%` applies only to the **teacher detector** (it can miss a detection). Our models always output a quad (`det%=100%`).

## Winners (by task)
- **Best cross‑dataset generalization (rev-last + rev-new):** `mobilenetv2_224_from256_clean_iter3`
- **Best on `rev-last` val:** `mobilenetv2_256_revlast` (slightly above the 224 fine‑tune)
- **Best on `rev-new` val:** `mobilenetv2_224_from256_clean_iter3`
- **Best on `val_clean_iter4_mix`:** `mobilenetv2_256_clean_iter3` (iter3)

## TFLite (top‑4) — float16 + int8 full‑quant

Artifacts:
- Exports: `checkpoints/tflite_top4/`
- Delegate report (XNNPACK): `checkpoints/tflite_top4/xnnpack_delegate_report.txt`
- Latency JSON (TFLite invoke-only, `threads=4`, `num_samples=2000`):
  - `checkpoints/tflite_top4/bench_revlast_val_invoke.json`
  - `checkpoints/tflite_top4/bench_revnew_val_invoke.json`
  - `checkpoints/tflite_top4/bench_val_clean_iter4_mix_invoke.json`

**Delegate check:** all models below are **100% delegated to XNNPACK** (no non-delegated builtin ops in the post‑delegate execution plan).

**Latency note:** timings are **`interpreter.invoke()` only** (no image decode / preprocessing), averaged over the first 2000 samples.

### TFLite float16 (coords9)

| model | tflite | size | XNNPACK | rev-last val | rev-new val | val_clean_iter4_mix |
|---|---|---:|---:|---:|---:|---:|
| `mobilenetv2_224_from256_clean_iter3` | `checkpoints/tflite_top4/mobilenetv2_224_from256_clean_iter3_float16.tflite` | 0.98 MB | ✅ | 3.263 ms | 3.683 ms | 3.572 ms |
| `mobilenetv2_256_clean_iter3` | `checkpoints/tflite_top4/mobilenetv2_256_clean_iter3_float16.tflite` | 0.98 MB | ✅ | 3.492 ms | 3.716 ms | 4.378 ms |
| `mobilenetv2_224_from256_revlast` | `checkpoints/tflite_top4/mobilenetv2_224_from256_revlast_float16.tflite` | 0.98 MB | ✅ | 3.144 ms | 3.687 ms | 3.642 ms |
| `mobilenetv2_256_revlast` | `checkpoints/tflite_top4/mobilenetv2_256_revlast_float16.tflite` | 0.98 MB | ✅ | 3.919 ms | 3.923 ms | 3.938 ms |

### TFLite int8 full‑quant (simcc logits)

| model | tflite | size | XNNPACK | rev-last val | rev-new val | val_clean_iter4_mix |
|---|---|---:|---:|---:|---:|---:|
| `mobilenetv2_224_from256_clean_iter3` | `checkpoints/tflite_top4/mobilenetv2_224_from256_clean_iter3_int8_full_simcc_int8io.tflite` | 0.82 MB | ✅ | 2.393 ms | 2.428 ms | 2.560 ms |
| `mobilenetv2_256_clean_iter3` | `checkpoints/tflite_top4/mobilenetv2_256_clean_iter3_int8_full_simcc_int8io.tflite` | 0.84 MB | ✅ | 2.922 ms | 2.458 ms | 4.200 ms |
| `mobilenetv2_224_from256_revlast` | `checkpoints/tflite_top4/mobilenetv2_224_from256_revlast_int8_full_simcc_int8io.tflite` | 0.82 MB | ✅ | 2.411 ms | 2.561 ms | 2.797 ms |
| `mobilenetv2_256_revlast` | `checkpoints/tflite_top4/mobilenetv2_256_revlast_int8_full_simcc_int8io.tflite` | 0.84 MB | ✅ | 2.908 ms | 2.486 ms | 3.246 ms |

### TFLite winners (accuracy)

- **Best float16 cross‑dataset (rev-last + rev-new):** `mobilenetv2_224_from256_clean_iter3` (worst‑case mIoU=0.9047)
- **Best int8 cross‑dataset (rev-last + rev-new):** `mobilenetv2_256_clean_iter3` (worst‑case mIoU=0.8900)

### TFLite float16 — accuracy (coords9)

**rev-last val**

| rank | model | Mean IoU | R@95 | R@99 | err mean px | err p95 px | err max px | cls_f1 | invoke mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `mobilenetv2_256_revlast` | 0.9798 | 96.2% | 43.6% | 1.49 | 2.40 | 240.27 | 0.999 | 3.919 ms |
| 2 | `mobilenetv2_224_from256_revlast` | 0.9783 | 95.8% | 37.2% | 1.39 | 2.32 | 209.55 | 1.000 | 3.144 ms |
| 3 | `mobilenetv2_224_from256_clean_iter3` | 0.9682 | 91.1% | 9.3% | 2.09 | 3.60 | 210.96 | 0.999 | 3.263 ms |
| 4 | `mobilenetv2_256_clean_iter3` | 0.9672 | 91.2% | 8.7% | 2.46 | 4.12 | 240.67 | 0.999 | 3.492 ms |

**rev-new val**

| rank | model | Mean IoU | R@95 | R@99 | err mean px | err p95 px | err max px | cls_f1 | invoke mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `mobilenetv2_224_from256_clean_iter3` | 0.9047 | 63.7% | 18.2% | 5.77 | 30.63 | 195.34 | 0.984 | 3.683 ms |
| 2 | `mobilenetv2_256_clean_iter3` | 0.8997 | 63.8% | 21.0% | 6.96 | 39.73 | 271.39 | 0.984 | 3.716 ms |
| 3 | `mobilenetv2_224_from256_revlast` | 0.8320 | 52.9% | 3.3% | 12.22 | 75.23 | 281.90 | 0.958 | 3.687 ms |
| 4 | `mobilenetv2_256_revlast` | 0.8301 | 53.2% | 3.2% | 13.89 | 86.90 | 283.14 | 0.966 | 3.923 ms |

**val_clean_iter4_mix**

| rank | model | Mean IoU | R@95 | R@99 | err mean px | err p95 px | err max px | cls_f1 | invoke mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `mobilenetv2_256_clean_iter3` | 0.9926 | 99.9% | 89.6% | 0.56 | 1.32 | 34.45 | 1.000 | 4.378 ms |
| 2 | `mobilenetv2_224_from256_clean_iter3` | 0.9908 | 99.8% | 69.8% | 0.59 | 1.40 | 31.62 | 1.000 | 3.572 ms |
| 3 | `mobilenetv2_224_from256_revlast` | 0.9730 | 93.7% | 10.0% | 1.62 | 3.50 | 190.70 | 0.998 | 3.642 ms |
| 4 | `mobilenetv2_256_revlast` | 0.9726 | 93.9% | 10.0% | 1.91 | 3.98 | 232.69 | 0.999 | 3.938 ms |

### TFLite int8 — accuracy (simcc logits, decode outside)

**rev-last val**

| rank | model | Mean IoU | R@95 | R@99 | err mean px | err p95 px | err max px | cls_f1 | invoke mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `mobilenetv2_224_from256_revlast` | 0.9773 | 95.2% | 35.3% | 1.47 | 2.43 | 209.28 | 0.999 | 2.411 ms |
| 2 | `mobilenetv2_256_revlast` | 0.9771 | 95.4% | 37.9% | 1.73 | 2.63 | 240.13 | 0.999 | 2.908 ms |
| 3 | `mobilenetv2_256_clean_iter3` | 0.9652 | 90.6% | 8.9% | 2.61 | 4.36 | 240.22 | 0.998 | 2.922 ms |
| 4 | `mobilenetv2_224_from256_clean_iter3` | 0.9639 | 89.8% | 10.0% | 2.35 | 3.95 | 210.77 | 0.998 | 2.393 ms |

**rev-new val**

| rank | model | Mean IoU | R@95 | R@99 | err mean px | err p95 px | err max px | cls_f1 | invoke mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `mobilenetv2_256_clean_iter3` | 0.8900 | 61.9% | 16.3% | 7.71 | 45.64 | 235.28 | 0.970 | 2.458 ms |
| 2 | `mobilenetv2_224_from256_clean_iter3` | 0.8887 | 61.0% | 13.8% | 6.93 | 40.70 | 204.65 | 0.971 | 2.428 ms |
| 3 | `mobilenetv2_224_from256_revlast` | 0.8298 | 52.4% | 3.3% | 12.34 | 75.50 | 279.25 | 0.957 | 2.561 ms |
| 4 | `mobilenetv2_256_revlast` | 0.8188 | 52.2% | 3.4% | 14.94 | 95.30 | 289.54 | 0.963 | 2.486 ms |

**val_clean_iter4_mix**

| rank | model | Mean IoU | R@95 | R@99 | err mean px | err p95 px | err max px | cls_f1 | invoke mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `mobilenetv2_256_clean_iter3` | 0.9901 | 99.6% | 66.5% | 0.71 | 1.63 | 73.82 | 1.000 | 4.200 ms |
| 2 | `mobilenetv2_224_from256_clean_iter3` | 0.9881 | 99.2% | 53.4% | 0.74 | 1.66 | 72.65 | 1.000 | 2.560 ms |
| 3 | `mobilenetv2_224_from256_revlast` | 0.9728 | 93.4% | 10.3% | 1.64 | 3.54 | 195.24 | 0.999 | 2.797 ms |
| 4 | `mobilenetv2_256_revlast` | 0.9717 | 93.2% | 10.7% | 2.00 | 4.14 | 235.39 | 0.999 | 3.246 ms |

## Overall leaderboard (rev-last + rev-new)
Ranked by **worst‑case Mean IoU** across `rev-last` val and `rev-new` val (so a model that collapses on one domain is penalized).

| rank | model | rev-last Mean IoU | rev-new Mean IoU | worst-case Mean IoU |
|---:|---|---:|---:|---:|
| 1 | `mobilenetv2_224_from256_clean_iter3` | 0.9682 | 0.9047 | **0.9047** |
| 2 | `mobilenetv2_256_clean_iter3` | 0.9672 | 0.8997 | **0.8997** |
| 3 | `mobilenetv2_224_from256_revlast` | 0.9784 | 0.8321 | **0.8321** |
| 4 | `mobilenetv2_256_revlast` | 0.9798 | 0.8301 | **0.8301** |

## Leaderboard — `val_clean_iter4_mix` (positives=5364, negatives=718)

| rank | model | Mean IoU | R@95 | R@99 | err mean px | err p95 px | err max px |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mobilenetv2_256_clean_iter3` | **0.9929** | 100.0% | 92.0% | 0.54 | 1.04 | 4.80 |
| 2 | `mobilenetv2_256_clean_iter4_mix` | 0.9925 | 99.8% | 81.8% | 0.55 | 1.22 | 64.87 |
| 3 | `mobilenetv2_256_clean_iter2` | 0.9910 | 99.8% | 78.3% | 0.69 | 1.22 | 204.75 |
| 4 | `mobilenetv2_224_from256_clean_iter3` | 0.9909 | 99.8% | 69.8% | 0.59 | 1.40 | 30.60 |
| 5 | `mobilenetv2_256_clean_iter1` | 0.9881 | 98.6% | 57.8% | 0.89 | 1.95 | 171.56 |
| 6 | `mobilenetv2_224_from256_revlast` | 0.9730 | 93.7% | 10.0% | 1.62 | 3.51 | 190.20 |
| 7 | `mobilenetv2_256_revlast` | 0.9726 | 93.8% | 10.0% | 1.91 | 3.99 | 232.66 |
| 8 | `teacher (det-only)` | 0.9657 | 90.7% | 5.9% | 9.48 | 20.47 | 840.67 |
| 9 | `mobilenetv2_256_best` | 0.9644 | 85.2% | 33.7% | 2.94 | 12.14 | 215.54 |

## Leaderboard — `rev-last` val (positives=8077, negatives=3114)

| rank | model | Mean IoU | R@95 | R@99 | err mean px | err p95 px | err max px |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mobilenetv2_256_revlast` | **0.9798** | 96.2% | 43.6% | 1.49 | 2.40 | 240.30 |
| 2 | `mobilenetv2_224_from256_revlast` | 0.9784 | 95.7% | 37.1% | 1.38 | 2.31 | 209.52 |
| 3 | `mobilenetv2_224_from256_clean_iter3` | 0.9682 | 91.2% | 9.4% | 2.09 | 3.60 | 210.97 |
| 4 | `mobilenetv2_256_clean_iter3` | 0.9672 | 91.2% | 8.6% | 2.46 | 4.10 | 240.68 |

## Leaderboard — `rev-new` val (positives=23444, negatives=3114)

| rank | model | Mean IoU | R@95 | R@99 | err mean px | err p95 px | err max px |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mobilenetv2_224_from256_clean_iter3` | **0.9047** | 63.7% | 18.2% | 5.77 | 30.57 | 195.35 |
| 2 | `mobilenetv2_256_clean_iter3` | 0.8997 | 63.8% | 21.0% | 6.97 | 39.91 | 266.76 |
| 3 | `mobilenetv2_224_from256_revlast` | 0.8321 | 52.9% | 3.2% | 12.21 | 74.95 | 281.87 |
| 4 | `mobilenetv2_256_revlast` | 0.8301 | 53.2% | 3.2% | 13.89 | 86.81 | 283.48 |

## Sources
- `evaluation_results/full_evaluation_iter1_on_val_iter4.csv`
- `evaluation_results/full_evaluation_iter2_on_val_iter4.csv`
- `evaluation_results/full_evaluation_iter3_on_val_iter4.csv`
- `evaluation_results/full_evaluation_iter4_on_val_iter4.csv`
- `evaluation_results/full_evaluation_best_on_val_iter4.csv`
- `evaluation_results/teacher_on_val_iter4_mix_pos.json`
- Local `evaluate.py` runs for cross‑dataset checks (`rev-last`, `rev-new`, and `val_clean_iter4_mix`)
