# e11 DEVELOPMENT corpora (seed 42, protocol v2)

Implements results/PREREG_confirmatory.md. Fit on train, test once; split seeds 42/123/456.
L_U detectors: V6/V3 label event restricted to USER_DEPENDENT_SCHEMA and trained on L_U;
sampling baselines on user-dependent turns only. L detectors: as in e6.


## user-specific label L_U (primary)

Session AUROC, mean over split seeds 42/123/456:

| corpus | V6 | V3 | V6-prior | ctx_lr | ctx_lr+fb | SelfCheckGPT | semantic_entropy | highconf_contradicted | neg_signal_fraction | dag_logreg | gcn | mean_confidence (oriented) | best graphon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 70B (dev) | 0.678 | 0.677 | 0.654 | 0.756 | 0.813 | 0.480 | 0.499 | 0.568 | 0.554 | 0.653 | 0.650 | 0.464 | 0.511 |
| 8B (dev) | 0.722 | 0.739 | 0.579 | 0.621 | 0.630 | 0.619 | 0.510 | 0.706 | 0.675 | 0.723 | 0.616 | 0.561 | 0.575 |

Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:

- **70B (dev)**: V6 - V6-prior: +0.023 [-0.007, +0.054]; V6 - ctx_lr: -0.073 [-0.132, -0.013]*; V6 - ctx_lr+fb: -0.132 [-0.187, -0.080]*; V6 - SelfCheckGPT: +0.201 [+0.123, +0.273]*; V6 - semantic_entropy: +0.181 [+0.123, +0.238]*; V6 - highconf_contradicted: +0.113 [+0.062, +0.167]*; V6 - V3: +0.001 [-0.004, +0.006]; ctx_lr+fb - ctx_lr: +0.059 [+0.025, +0.095]*
- **8B (dev)**: V6 - V6-prior: +0.131 [+0.072, +0.194]*; V6 - ctx_lr: +0.090 [-0.031, +0.214]; V6 - ctx_lr+fb: +0.078 [-0.037, +0.202]; V6 - SelfCheckGPT: +0.078 [-0.080, +0.229]; V6 - semantic_entropy: +0.196 [+0.092, +0.290]*; V6 - highconf_contradicted: +0.001 [-0.112, +0.110]; V6 - V3: -0.027 [-0.049, -0.006]*; ctx_lr+fb - ctx_lr: +0.011 [-0.062, +0.084]

Data efficiency (Amendment 1, descriptive): test AUROC vs labelled training sessions

| corpus | detector | 25 | 50 | 100 | 200 | all train |
|---|---|---|---|---|---|---|
| 70B (dev) | V6 | 0.628 | 0.648 | 0.663 | 0.671 | 0.678 |
| 70B (dev) | ctx_lr | 0.626 | 0.668 | 0.699 | 0.728 | 0.756 |
| 70B (dev) | ctx_lr+fb | 0.649 | 0.710 | 0.729 | 0.774 | 0.813 |

## composite label L (secondary)

Session AUROC, mean over split seeds 42/123/456:

| corpus | V6 | V3 | V6-prior | ctx_lr | ctx_lr+fb | SelfCheckGPT | semantic_entropy | highconf_contradicted | neg_signal_fraction | dag_logreg | gcn | mean_confidence (oriented) | best graphon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 70B (dev) | 0.774 | 0.775 | 0.747 | 0.772 | 0.791 | 0.504 | 0.499 | 0.711 | 0.686 | 0.705 | 0.679 | 0.680 | 0.445 |
| 8B (dev) | 0.734 | 0.728 | 0.697 | 0.652 | 0.660 | 0.669 | 0.506 | 0.596 | 0.569 | 0.636 | 0.632 | 0.476 | 0.557 |

Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:

- **70B (dev)**: V6 - V6-prior: +0.028 [-0.012, +0.065]; V6 - ctx_lr: +0.000 [-0.051, +0.051]; V6 - ctx_lr+fb: -0.018 [-0.066, +0.030]; V6 - SelfCheckGPT: +0.268 [+0.190, +0.345]*; V6 - semantic_entropy: +0.273 [+0.222, +0.324]*; V6 - highconf_contradicted: +0.063 [+0.009, +0.120]*; V6 - V3: -0.003 [-0.012, +0.005]; ctx_lr+fb - ctx_lr: +0.018 [-0.023, +0.060]
- **8B (dev)**: V6 - V6-prior: +0.045 [-0.030, +0.119]; V6 - ctx_lr: +0.087 [-0.036, +0.209]; V6 - ctx_lr+fb: +0.070 [-0.037, +0.180]; V6 - SelfCheckGPT: +0.060 [-0.064, +0.186]; V6 - semantic_entropy: +0.224 [+0.131, +0.307]*; V6 - highconf_contradicted: +0.132 [+0.047, +0.216]*; V6 - V3: +0.001 [-0.026, +0.028]; ctx_lr+fb - ctx_lr: +0.016 [-0.042, +0.074]

Data efficiency (Amendment 1, descriptive): test AUROC vs labelled training sessions

| corpus | detector | 25 | 50 | 100 | 200 | all train |
|---|---|---|---|---|---|---|
| 70B (dev) | V6 | 0.715 | 0.723 | 0.755 | 0.763 | 0.774 |
| 70B (dev) | ctx_lr | 0.682 | 0.713 | 0.724 | 0.769 | 0.772 |
| 70B (dev) | ctx_lr+fb | 0.684 | 0.711 | 0.732 | 0.775 | 0.791 |

## Per-commitment check of Corollary 1 (user-dependent commitments; descriptive)

| corpus | n | wrong | all resamples agree | e1 = P(disagree|wrong) | e0 = P(disagree|right) | Cor. 1 bound | AUROC disagreement | AUROC one feedback bit |
|---|---|---|---|---|---|---|---|---|
| 70B (dev) | 1092 | 0.508 | 0.921 | 0.079 | 0.078 | 0.576 | 0.500 | 0.764 |
| 8B (dev) | 384 | 0.406 | 0.487 | 0.538 | 0.496 | 0.884 | 0.536 | 0.782 |

## Amendment 2: rules-aware analyses on L_U (oracle information: the simulator's rules and hidden-fact prior; not deployable)

| corpus | eta0 (Bayes, no feedback) | etaF (Bayes, feedback) | V6-session | LR-session | LR-session+fb |
|---|---|---|---|---|---|
| 70B (dev) | 0.908 | 0.954 | 0.904 | 0.887 | 0.884 |
| 8B (dev) | 0.901 | 0.975 | 0.956 | 0.858 | 0.856 |

Differences per corpus (session-clustered):

- **70B (dev)**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.045 [+0.023, +0.070]*; V6-session - LR-session+fb: +0.023 [-0.013, +0.059]; V6-session - eta0 (Bayes, no feedback): -0.005 [-0.036, +0.026]; LR-session+fb - LR-session: -0.005 [-0.043, +0.030]
- **8B (dev)**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.073 [+0.033, +0.122]*; V6-session - LR-session+fb: +0.087 [+0.020, +0.164]*; V6-session - eta0 (Bayes, no feedback): +0.048 [-0.008, +0.111]; LR-session+fb - LR-session: +0.009 [-0.059, +0.066]

runtime 199s
