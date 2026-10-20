# e12 EXPLORATORY (not pre-registered; requested in review)

Same corpora, splits (42/123/456, 70/30) and bootstrap as e11.

## Healthcare, confirmatory corpora

> 70B: not found; skipped.
### L_U: session AUROC (mean over split seeds)

| corpus | ctx_lr | ctx_lr+fb | intake_lr | intake_lr+fb | V6 |
|---|---|---|---|---|---|
| 8B | 0.686 | 0.725 | 0.772 | 0.782 | 0.672 |
| qwen72b | 0.715 | 0.748 | 0.723 | 0.749 | 0.650 |
| gpt-4o-mini | 0.635 | 0.668 | 0.678 | 0.717 | 0.636 |
| mistral-small-24b | 0.556 | 0.682 | 0.585 | 0.690 | 0.666 |

Pooled over 4 corpora (stratified cluster bootstrap):

- intake_lr - ctx_lr: +0.042 [+0.012, +0.068]
- intake_lr+fb - intake_lr: +0.045 [+0.032, +0.059]
- intake_lr+fb - ctx_lr+fb: +0.029 [+0.002, +0.054]
- V6 - intake_lr+fb: -0.079 [-0.113, -0.043]
- AUROC(intake_lr): +0.689 [+0.658, +0.715]
- AUROC(intake_lr+fb): +0.735 [+0.704, +0.759]

### L: session AUROC (mean over split seeds)

| corpus | ctx_lr | ctx_lr+fb | intake_lr | intake_lr+fb | V6 |
|---|---|---|---|---|---|
| 8B | 0.744 | 0.810 | 0.875 | 0.888 | 0.829 |
| qwen72b | 0.853 | 0.879 | 0.897 | 0.910 | 0.881 |
| gpt-4o-mini | 0.806 | 0.809 | 0.904 | 0.907 | 0.849 |
| mistral-small-24b | 0.840 | 0.872 | 0.891 | 0.906 | 0.845 |

Pooled over 4 corpora (stratified cluster bootstrap):

- intake_lr - ctx_lr: +0.081 [+0.061, +0.105]
- intake_lr+fb - intake_lr: +0.011 [+0.005, +0.017]
- intake_lr+fb - ctx_lr+fb: +0.061 [+0.044, +0.080]
- V6 - intake_lr+fb: -0.052 [-0.077, -0.031]
- AUROC(intake_lr): +0.892 [+0.874, +0.909]
- AUROC(intake_lr+fb): +0.903 [+0.885, +0.919]

### Per-commitment q_t (V6, L_U fit; confident user-dependent commitments, test splits)

| corpus | n per split | wrong | mean q | AUROC(q) | ECE(q) |
|---|---|---|---|---|---|
| 8B | 159 | 0.427 | 0.305 | 0.748 | 0.160 |
| qwen72b | 262 | 0.416 | 0.424 | 0.776 | 0.084 |
| gpt-4o-mini | 246 | 0.394 | 0.440 | 0.753 | 0.108 |
| mistral-small-24b | 319 | 0.406 | 0.448 | 0.797 | 0.093 |

runtime 133s
