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

### L_U label-budget curve, intake-aware LRs (same subsets as e11)

| corpus | detector | 25 | 50 | 100 | 200 | all |
|---|---|---|---|---|---|---|
| 8B | intake_lr | 0.636 | 0.684 | 0.716 | 0.757 | 0.772 |
| 8B | intake_lr+fb | 0.633 | 0.687 | 0.726 | 0.770 | 0.782 |
| qwen72b | intake_lr | 0.610 | 0.642 | 0.666 | 0.707 | 0.723 |
| qwen72b | intake_lr+fb | 0.621 | 0.660 | 0.689 | 0.739 | 0.749 |
| gpt-4o-mini | intake_lr | 0.593 | 0.611 | 0.640 | 0.665 | 0.678 |
| gpt-4o-mini | intake_lr+fb | 0.600 | 0.628 | 0.650 | 0.693 | 0.717 |
| mistral-small-24b | intake_lr | 0.543 | 0.570 | 0.567 | 0.582 | 0.585 |
| mistral-small-24b | intake_lr+fb | 0.559 | 0.600 | 0.621 | 0.659 | 0.690 |

### Per-commitment risk (V6 fit on L_U; confident user-dependent commitments, test splits)

| corpus | n per split | wrong | mean q | AUROC q_t | ECE q_t | AUROC prior | AUROC reaction | AUROC disagreement | AUROC stated conf. |
|---|---|---|---|---|---|---|---|---|---|
| 8B | 159 | 0.427 | 0.305 | 0.748 | 0.160 | 0.563 | 0.755 | 0.577 | 0.545 |
| qwen72b | 262 | 0.416 | 0.424 | 0.776 | 0.084 | 0.591 | 0.756 | 0.520 | 0.552 |
| gpt-4o-mini | 246 | 0.394 | 0.440 | 0.753 | 0.108 | 0.554 | 0.740 | 0.538 | 0.507 |
| mistral-small-24b | 319 | 0.406 | 0.448 | 0.797 | 0.093 | 0.588 | 0.761 | 0.564 | 0.484 |

runtime 241s
