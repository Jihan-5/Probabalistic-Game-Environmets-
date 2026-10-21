# e20: LLM judge (gpt-4o-mini, zero-shot, reads intake + commitments + reactions) vs V7 (exploratory)

## L_U

| corpus | V7 | intake_lr+fb | SelfCheckGPT | llm_judge |
|---|---|---|---|---|
| 70B | 0.902 | 0.743 | 0.492 | 0.645 |
| 8B | 0.889 | 0.782 | 0.610 | 0.611 |
| qwen72b | 0.810 | 0.749 | 0.559 | 0.647 |
| gpt-4o-mini | 0.836 | 0.717 | 0.576 | 0.612 |
| mistral-small-24b | 0.690 | 0.690 | 0.520 | 0.627 |

- 70B: V7 - llm_judge +0.257 [+0.199, +0.318]; V7 - intake_lr+fb +0.160 [+0.106, +0.216]; V7 - SelfCheckGPT +0.408 [+0.348, +0.472]; llm_judge - SelfCheckGPT +0.151 [+0.073, +0.227]
- 8B: V7 - llm_judge +0.275 [+0.211, +0.339]; V7 - intake_lr+fb +0.108 [+0.061, +0.158]; V7 - SelfCheckGPT +0.277 [+0.199, +0.351]; llm_judge - SelfCheckGPT +0.003 [-0.079, +0.084]
- qwen72b: V7 - llm_judge +0.162 [+0.103, +0.221]; V7 - intake_lr+fb +0.060 [+0.010, +0.111]; V7 - SelfCheckGPT +0.249 [+0.172, +0.322]; llm_judge - SelfCheckGPT +0.086 [+0.011, +0.163]
- gpt-4o-mini: V7 - llm_judge +0.222 [+0.157, +0.283]; V7 - intake_lr+fb +0.120 [+0.068, +0.176]; V7 - SelfCheckGPT +0.257 [+0.199, +0.315]; llm_judge - SelfCheckGPT +0.034 [-0.028, +0.094]
- mistral-small-24b: V7 - llm_judge +0.069 [+0.004, +0.136]; V7 - intake_lr+fb +0.007 [-0.057, +0.067]; V7 - SelfCheckGPT +0.177 [+0.105, +0.249]; llm_judge - SelfCheckGPT +0.108 [+0.032, +0.184]

Pooled over 5 corpora:
- V7 - llm_judge: +0.197 [+0.169, +0.224]
- llm_judge - SelfCheckGPT: +0.077 [+0.044, +0.110]
- AUROC(llm_judge): +0.628 [+0.601, +0.656]

## L

| corpus | V7 | intake_lr+fb | SelfCheckGPT | llm_judge |
|---|---|---|---|---|
| 70B | 0.914 | 0.871 | 0.520 | 0.649 |
| 8B | 0.918 | 0.888 | 0.641 | 0.633 |
| qwen72b | 0.944 | 0.910 | 0.634 | 0.693 |
| gpt-4o-mini | 0.917 | 0.907 | 0.730 | 0.678 |
| mistral-small-24b | 0.924 | 0.906 | 0.698 | 0.727 |

- 70B: V7 - llm_judge +0.267 [+0.209, +0.326]; V7 - intake_lr+fb +0.047 [+0.006, +0.087]; V7 - SelfCheckGPT +0.396 [+0.330, +0.459]; llm_judge - SelfCheckGPT +0.129 [+0.050, +0.210]
- 8B: V7 - llm_judge +0.286 [+0.223, +0.348]; V7 - intake_lr+fb +0.028 [-0.009, +0.064]; V7 - SelfCheckGPT +0.275 [+0.217, +0.339]; llm_judge - SelfCheckGPT -0.011 [-0.097, +0.073]
- qwen72b: V7 - llm_judge +0.250 [+0.190, +0.311]; V7 - intake_lr+fb +0.034 [+0.006, +0.064]; V7 - SelfCheckGPT +0.313 [+0.246, +0.379]; llm_judge - SelfCheckGPT +0.062 [-0.022, +0.152]
- gpt-4o-mini: V7 - llm_judge +0.237 [+0.165, +0.312]; V7 - intake_lr+fb +0.007 [-0.029, +0.046]; V7 - SelfCheckGPT +0.181 [+0.118, +0.247]; llm_judge - SelfCheckGPT -0.056 [-0.140, +0.032]
- mistral-small-24b: V7 - llm_judge +0.192 [+0.134, +0.251]; V7 - intake_lr+fb +0.018 [-0.019, +0.057]; V7 - SelfCheckGPT +0.221 [+0.155, +0.285]; llm_judge - SelfCheckGPT +0.030 [-0.050, +0.110]

Pooled over 5 corpora:
- V7 - llm_judge: +0.247 [+0.217, +0.276]
- llm_judge - SelfCheckGPT: +0.031 [-0.005, +0.070]
- AUROC(llm_judge): +0.676 [+0.648, +0.704]

## per commitment (confident, user-dependent; mean over split seeds)

- 70B: judge 0.755, V7 0.834
- 8B: judge 0.730, V7 0.799
- qwen72b: judge 0.759, V7 0.778
- gpt-4o-mini: judge 0.751, V7 0.766
- mistral-small-24b: judge 0.742, V7 0.806

runtime 300s
