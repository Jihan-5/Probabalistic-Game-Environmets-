# e17: stronger supervised comparators and V7 without reactions (EXPLORATORY; L_U)

> 70B: not found; skipped.
| corpus | V7 | V7-prior | intake_lr+fb | intake_lr+fb (cv) | dep_lr+fb (cv) | intake_gbm+fb |
|---|---|---|---|---|---|---|
| 8B | 0.889 | 0.819 | 0.782 | 0.808 | 0.848 | 0.802 |
| qwen72b | 0.810 | 0.737 | 0.749 | 0.767 | 0.801 | 0.767 |
| gpt-4o-mini | 0.836 | 0.739 | 0.717 | 0.726 | 0.815 | 0.726 |
| mistral-small-24b | 0.690 | 0.569 | 0.690 | 0.703 | 0.762 | 0.728 |

- 8B: V7 - V7-prior +0.070 [+0.041, +0.100]; V7 - intake_lr+fb +0.108 [+0.061, +0.158]; V7 - intake_lr+fb (cv) +0.081 [+0.036, +0.126]; V7 - dep_lr+fb (cv) +0.039 [+0.003, +0.076]; V7 - intake_gbm+fb +0.089 [+0.041, +0.139]
- qwen72b: V7 - V7-prior +0.072 [+0.042, +0.103]; V7 - intake_lr+fb +0.060 [+0.010, +0.111]; V7 - intake_lr+fb (cv) +0.044 [-0.005, +0.094]; V7 - dep_lr+fb (cv) +0.009 [-0.033, +0.049]; V7 - intake_gbm+fb +0.041 [-0.010, +0.093]
- gpt-4o-mini: V7 - V7-prior +0.099 [+0.067, +0.134]; V7 - intake_lr+fb +0.120 [+0.068, +0.176]; V7 - intake_lr+fb (cv) +0.110 [+0.062, +0.163]; V7 - dep_lr+fb (cv) +0.021 [-0.016, +0.059]; V7 - intake_gbm+fb +0.110 [+0.061, +0.160]
- mistral-small-24b: V7 - V7-prior +0.121 [+0.083, +0.160]; V7 - intake_lr+fb +0.007 [-0.057, +0.067]; V7 - intake_lr+fb (cv) -0.000 [-0.062, +0.059]; V7 - dep_lr+fb (cv) -0.063 [-0.115, -0.010]; V7 - intake_gbm+fb -0.032 [-0.084, +0.021]

Pooled over 4 corpora:

- V7 - V7-prior: +0.090 [+0.074, +0.108]
- V7 - intake_lr+fb: +0.071 [+0.047, +0.101]
- V7 - intake_lr+fb (cv): +0.055 [+0.033, +0.085]
- V7 - dep_lr+fb (cv): -0.000 [-0.019, +0.022]
- V7 - intake_gbm+fb: +0.050 [+0.027, +0.076]

runtime 238s
