# e16: Amendment 6 (V7, frozen at d32ea33) on the confirmatory corpora

> 70B: not found; skipped.
## healthcare: L_U

| corpus | V6 | V7 | ctx_lr | ctx_lr+fb | intake_lr+fb | SelfCheckGPT | semantic_entropy |
|---|---|---|---|---|---|---|---|
| 8B | 0.672 | 0.889 | 0.686 | 0.725 | 0.782 | 0.610 | 0.533 |
| qwen72b | 0.650 | 0.810 | 0.715 | 0.748 | 0.749 | 0.559 | 0.485 |
| gpt-4o-mini | 0.636 | 0.836 | 0.635 | 0.668 | 0.717 | 0.576 | 0.486 |
| mistral-small-24b | 0.666 | 0.690 | 0.556 | 0.682 | 0.690 | 0.520 | 0.474 |

- 8B: V7 - V6 +0.217 [+0.160, +0.280]; V7 - intake_lr+fb +0.108 [+0.061, +0.158]; V7 - ctx_lr+fb +0.161 [+0.103, +0.220]; V7 - SelfCheckGPT +0.277 [+0.199, +0.351]; V7 - semantic_entropy +0.356 [+0.294, +0.415]
- qwen72b: V7 - V6 +0.163 [+0.108, +0.216]; V7 - intake_lr+fb +0.060 [+0.010, +0.111]; V7 - ctx_lr+fb +0.060 [+0.009, +0.110]; V7 - SelfCheckGPT +0.249 [+0.172, +0.322]; V7 - semantic_entropy +0.322 [+0.270, +0.375]
- gpt-4o-mini: V7 - V6 +0.198 [+0.142, +0.254]; V7 - intake_lr+fb +0.120 [+0.068, +0.176]; V7 - ctx_lr+fb +0.169 [+0.118, +0.224]; V7 - SelfCheckGPT +0.257 [+0.199, +0.315]; V7 - semantic_entropy +0.350 [+0.303, +0.394]
- mistral-small-24b: V7 - V6 +0.027 [-0.030, +0.084]; V7 - intake_lr+fb +0.007 [-0.057, +0.067]; V7 - ctx_lr+fb +0.016 [-0.042, +0.074]; V7 - SelfCheckGPT +0.177 [+0.105, +0.249]; V7 - semantic_entropy +0.223 [+0.166, +0.281]

Pooled over 4 corpora (stratified cluster bootstrap):

- H7 V7 - V6: +0.150 [+0.123, +0.180]
- H8 V7 - intake_lr+fb: +0.071 [+0.047, +0.101]
- H8' V7 - ctx_lr+fb: +0.100 [+0.074, +0.129]
- -- V7 - SelfCheckGPT: +0.240 [+0.206, +0.276]
- -- V7 - semantic_entropy: +0.312 [+0.284, +0.339]
- -- AUROC(V7): +0.806 [+0.784, +0.828]
- -- AUROC(intake_lr+fb): +0.735 [+0.704, +0.759]

## healthcare: L

| corpus | V6 | V7 | ctx_lr | ctx_lr+fb | intake_lr+fb | SelfCheckGPT | semantic_entropy |
|---|---|---|---|---|---|---|---|
| 8B | 0.829 | 0.918 | 0.744 | 0.810 | 0.888 | 0.641 | 0.572 |
| qwen72b | 0.881 | 0.944 | 0.853 | 0.879 | 0.910 | 0.634 | 0.530 |
| gpt-4o-mini | 0.849 | 0.917 | 0.806 | 0.809 | 0.907 | 0.730 | 0.571 |
| mistral-small-24b | 0.845 | 0.924 | 0.840 | 0.872 | 0.906 | 0.698 | 0.524 |

- 8B: V7 - V6 +0.089 [+0.056, +0.124]; V7 - intake_lr+fb +0.028 [-0.009, +0.064]; V7 - ctx_lr+fb +0.107 [+0.068, +0.151]; V7 - SelfCheckGPT +0.275 [+0.217, +0.339]; V7 - semantic_entropy +0.345 [+0.285, +0.407]
- qwen72b: V7 - V6 +0.063 [+0.038, +0.090]; V7 - intake_lr+fb +0.034 [+0.006, +0.064]; V7 - ctx_lr+fb +0.066 [+0.035, +0.099]; V7 - SelfCheckGPT +0.313 [+0.246, +0.379]; V7 - semantic_entropy +0.414 [+0.372, +0.454]
- gpt-4o-mini: V7 - V6 +0.069 [+0.042, +0.099]; V7 - intake_lr+fb +0.007 [-0.029, +0.046]; V7 - ctx_lr+fb +0.106 [+0.054, +0.159]; V7 - SelfCheckGPT +0.181 [+0.118, +0.247]; V7 - semantic_entropy +0.339 [+0.281, +0.392]
- mistral-small-24b: V7 - V6 +0.086 [+0.057, +0.119]; V7 - intake_lr+fb +0.018 [-0.019, +0.057]; V7 - ctx_lr+fb +0.056 [+0.018, +0.096]; V7 - SelfCheckGPT +0.221 [+0.155, +0.285]; V7 - semantic_entropy +0.401 [+0.348, +0.451]

Pooled over 4 corpora (stratified cluster bootstrap):

- H7 V7 - V6: +0.075 [+0.062, +0.091]
- H8 V7 - intake_lr+fb: +0.023 [+0.004, +0.039]
- H8' V7 - ctx_lr+fb: +0.084 [+0.062, +0.105]
- -- V7 - SelfCheckGPT: +0.250 [+0.215, +0.280]
- -- V7 - semantic_entropy: +0.377 [+0.350, +0.401]
- -- AUROC(V7): +0.926 [+0.910, +0.938]
- -- AUROC(intake_lr+fb): +0.903 [+0.885, +0.919]

## healthcare: L_U label budget (descriptive)

- 8B: V7@100 0.767, V7@200 0.827, V7@25 0.651, V7@50 0.714, ctx_lr+fb@100 0.657, ctx_lr+fb@200 0.685, ctx_lr+fb@25 0.572, ctx_lr+fb@50 0.617, intake_lr+fb@100 0.726, intake_lr+fb@200 0.770, intake_lr+fb@25 0.633, intake_lr+fb@50 0.687
- qwen72b: V7@100 0.675, V7@200 0.706, V7@25 0.647, V7@50 0.636, ctx_lr+fb@100 0.669, ctx_lr+fb@200 0.719, ctx_lr+fb@25 0.614, ctx_lr+fb@50 0.651, intake_lr+fb@100 0.689, intake_lr+fb@200 0.739, intake_lr+fb@25 0.621, intake_lr+fb@50 0.660
- gpt-4o-mini: V7@100 0.723, V7@200 0.793, V7@25 0.601, V7@50 0.600, ctx_lr+fb@100 0.622, ctx_lr+fb@200 0.647, ctx_lr+fb@25 0.574, ctx_lr+fb@50 0.607, intake_lr+fb@100 0.650, intake_lr+fb@200 0.693, intake_lr+fb@25 0.600, intake_lr+fb@50 0.628
- mistral-small-24b: V7@100 0.672, V7@200 0.703, V7@25 0.630, V7@50 0.625, ctx_lr+fb@100 0.609, ctx_lr+fb@200 0.648, ctx_lr+fb@25 0.571, ctx_lr+fb@50 0.590, intake_lr+fb@100 0.621, intake_lr+fb@200 0.659, intake_lr+fb@25 0.559, intake_lr+fb@50 0.600

## healthcare: per-commitment q_t of V7

- 8B: AUROC 0.801, ECE 0.065 (n per split 159)
- qwen72b: AUROC 0.775, ECE 0.059 (n per split 262)
- gpt-4o-mini: AUROC 0.765, ECE 0.077 (n per split 246)
- mistral-small-24b: AUROC 0.803, ECE 0.049 (n per split 319)

> gpt-4o-mini (travel): not found; skipped.
> 8B (travel): not found; skipped.
runtime 648s
