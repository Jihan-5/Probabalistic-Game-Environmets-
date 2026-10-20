# e11 confirmatory corpora (seed 7, protocol v2fb)

Implements results/PREREG_confirmatory.md. Fit on train, test once; split seeds 42/123/456.
L_U detectors: V6/V3 label event restricted to USER_DEPENDENT_SCHEMA and trained on L_U;
sampling baselines on user-dependent turns only. L detectors: as in e6.

> 70B: `data/trajectories/healthcare_confirm/70B_7.jsonl` not found; skipped.

## user-specific label L_U (primary)

Session AUROC, mean over split seeds 42/123/456:

| corpus | V6 | V3 | V6-prior | ctx_lr | ctx_lr+fb | SelfCheckGPT | semantic_entropy | highconf_contradicted | neg_signal_fraction | dag_logreg | gcn | mean_confidence (oriented) | best graphon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8B | 0.672 | 0.669 | 0.591 | 0.686 | 0.725 | 0.610 | 0.533 | 0.614 | 0.536 | 0.647 | 0.574 | 0.512 | 0.531 |
| qwen72b | 0.650 | 0.654 | 0.593 | 0.715 | 0.748 | 0.559 | 0.485 | 0.628 | 0.579 | 0.653 | 0.663 | 0.522 | 0.506 |
| gpt-4o-mini | 0.636 | 0.635 | 0.575 | 0.635 | 0.668 | 0.576 | 0.486 | 0.576 | 0.595 | 0.609 | 0.597 | 0.465 | 0.521 |
| mistral-small-24b | 0.666 | 0.673 | 0.577 | 0.556 | 0.682 | 0.520 | 0.474 | 0.637 | 0.624 | 0.596 | 0.569 | 0.454 | 0.409 |

Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:

- **8B**: V6 - V6-prior: +0.079 [+0.042, +0.116]*; V6 - ctx_lr: -0.016 [-0.082, +0.052]; V6 - ctx_lr+fb: -0.056 [-0.111, -0.001]*; V6 - SelfCheckGPT: +0.060 [-0.015, +0.134]; V6 - semantic_entropy: +0.139 [+0.067, +0.209]*; V6 - highconf_contradicted: +0.055 [-0.007, +0.117]; V6 - V3: +0.001 [-0.008, +0.011]; ctx_lr+fb - ctx_lr: +0.040 [+0.002, +0.079]*
- **qwen72b**: V6 - V6-prior: +0.057 [+0.023, +0.091]*; V6 - ctx_lr: -0.070 [-0.137, -0.002]*; V6 - ctx_lr+fb: -0.103 [-0.163, -0.044]*; V6 - SelfCheckGPT: +0.086 [+0.014, +0.162]*; V6 - semantic_entropy: +0.159 [+0.096, +0.220]*; V6 - highconf_contradicted: +0.019 [-0.027, +0.063]; V6 - V3: -0.007 [-0.016, +0.003]; ctx_lr+fb - ctx_lr: +0.033 [-0.009, +0.075]
- **gpt-4o-mini**: V6 - V6-prior: +0.060 [+0.026, +0.095]*; V6 - ctx_lr: +0.005 [-0.070, +0.079]; V6 - ctx_lr+fb: -0.029 [-0.099, +0.040]; V6 - SelfCheckGPT: +0.058 [+0.004, +0.111]*; V6 - semantic_entropy: +0.151 [+0.090, +0.215]*; V6 - highconf_contradicted: +0.058 [-0.002, +0.114]; V6 - V3: -0.000 [-0.008, +0.007]; ctx_lr+fb - ctx_lr: +0.034 [-0.004, +0.075]
- **mistral-small-24b**: V6 - V6-prior: +0.090 [+0.052, +0.127]*; V6 - ctx_lr: +0.115 [+0.042, +0.187]*; V6 - ctx_lr+fb: -0.011 [-0.073, +0.051]; V6 - SelfCheckGPT: +0.150 [+0.079, +0.222]*; V6 - semantic_entropy: +0.197 [+0.138, +0.255]*; V6 - highconf_contradicted: +0.032 [-0.019, +0.083]; V6 - V3: -0.005 [-0.011, +0.001]; ctx_lr+fb - ctx_lr: +0.126 [+0.077, +0.175]*

Data efficiency (Amendment 1, descriptive): test AUROC vs labelled training sessions

| corpus | detector | 25 | 50 | 100 | 200 | all train |
|---|---|---|---|---|---|---|
| 8B | V6 | 0.606 | 0.622 | 0.648 | 0.653 | 0.672 |
| 8B | ctx_lr | 0.579 | 0.604 | 0.641 | 0.672 | 0.686 |
| 8B | ctx_lr+fb | 0.572 | 0.617 | 0.657 | 0.685 | 0.725 |
| qwen72b | V6 | 0.644 | 0.645 | 0.648 | 0.648 | 0.650 |
| qwen72b | ctx_lr | 0.588 | 0.637 | 0.636 | 0.697 | 0.715 |
| qwen72b | ctx_lr+fb | 0.614 | 0.651 | 0.669 | 0.719 | 0.748 |
| gpt-4o-mini | V6 | 0.582 | 0.598 | 0.612 | 0.629 | 0.636 |
| gpt-4o-mini | ctx_lr | 0.559 | 0.572 | 0.595 | 0.612 | 0.635 |
| gpt-4o-mini | ctx_lr+fb | 0.574 | 0.607 | 0.622 | 0.647 | 0.668 |
| mistral-small-24b | V6 | 0.627 | 0.638 | 0.659 | 0.659 | 0.666 |
| mistral-small-24b | ctx_lr | 0.521 | 0.529 | 0.544 | 0.556 | 0.556 |
| mistral-small-24b | ctx_lr+fb | 0.571 | 0.590 | 0.609 | 0.648 | 0.682 |

Pooled over 4 corpora (stratified cluster bootstrap):

| id | quantity | estimate [95% CI] | licensed claim |
|---|---|---|---|
| H1 | AUROC(SelfCheckGPT) | 0.566 [0.535, 0.595] | sampling detectors are near chance on user-specific hallucination |
| H1 | AUROC(semantic_entropy) | 0.494 [0.479, 0.510] | sampling detectors are near chance on user-specific hallucination |
| H2 | V6 - V6-prior | +0.072 [+0.054, +0.090]* | the user's implicit reactions add detectable signal beyond population base rates |
| H3 | V6 - ctx_lr | +0.008 [-0.028, +0.045] | context alone matches PGE on this label |
| H4 | V6 - SelfCheckGPT | +0.090 [+0.056, +0.123]* | PGE beats sampling-consistency detectors |
| H4 | V6 - semantic_entropy | +0.162 [+0.127, +0.195]* | PGE beats sampling-consistency detectors |
| H2b | ctx_lr+fb - ctx_lr | +0.058 [+0.038, +0.080]* | the user's reactions add detectable signal beyond the commitments themselves (supervised detector) |
| H4b | ctx_lr+fb - SelfCheckGPT | +0.139 [+0.101, +0.176]* | detectors on PGE observables beat sampling-consistency detectors |
| H4b | ctx_lr+fb - semantic_entropy | +0.211 [+0.179, +0.242]* | detectors on PGE observables beat sampling-consistency detectors |
| sec. | V6 - ctx_lr+fb | -0.050 [-0.084, -0.019]* | reported as is |
| sec. | V6 - V3 | -0.002 [-0.007, +0.002] | reported as is |

## composite label L (secondary)

Session AUROC, mean over split seeds 42/123/456:

| corpus | V6 | V3 | V6-prior | ctx_lr | ctx_lr+fb | SelfCheckGPT | semantic_entropy | highconf_contradicted | neg_signal_fraction | dag_logreg | gcn | mean_confidence (oriented) | best graphon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8B | 0.829 | 0.828 | 0.717 | 0.744 | 0.810 | 0.641 | 0.572 | 0.785 | 0.713 | 0.783 | 0.708 | 0.642 | 0.533 |
| qwen72b | 0.881 | 0.883 | 0.796 | 0.853 | 0.879 | 0.634 | 0.530 | 0.846 | 0.826 | 0.863 | 0.812 | 0.718 | 0.448 |
| gpt-4o-mini | 0.849 | 0.857 | 0.815 | 0.806 | 0.809 | 0.730 | 0.571 | 0.770 | 0.761 | 0.810 | 0.771 | 0.756 | 0.419 |
| mistral-small-24b | 0.845 | 0.852 | 0.754 | 0.840 | 0.872 | 0.698 | 0.524 | 0.821 | 0.812 | 0.821 | 0.764 | 0.679 | 0.414 |

Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:

- **8B**: V6 - V6-prior: +0.113 [+0.074, +0.152]*; V6 - ctx_lr: +0.086 [+0.028, +0.145]*; V6 - ctx_lr+fb: +0.018 [-0.022, +0.058]; V6 - SelfCheckGPT: +0.186 [+0.115, +0.257]*; V6 - semantic_entropy: +0.255 [+0.187, +0.322]*; V6 - highconf_contradicted: +0.044 [+0.009, +0.080]*; V6 - V3: +0.002 [-0.004, +0.007]; ctx_lr+fb - ctx_lr: +0.067 [+0.037, +0.099]*
- **qwen72b**: V6 - V6-prior: +0.085 [+0.054, +0.117]*; V6 - ctx_lr: +0.031 [-0.002, +0.067]; V6 - ctx_lr+fb: +0.003 [-0.026, +0.033]; V6 - SelfCheckGPT: +0.250 [+0.181, +0.320]*; V6 - semantic_entropy: +0.352 [+0.301, +0.399]*; V6 - highconf_contradicted: +0.035 [+0.001, +0.069]*; V6 - V3: -0.003 [-0.007, +0.001]; ctx_lr+fb - ctx_lr: +0.028 [+0.004, +0.054]*
- **gpt-4o-mini**: V6 - V6-prior: +0.037 [+0.014, +0.062]*; V6 - ctx_lr: +0.042 [-0.004, +0.092]; V6 - ctx_lr+fb: +0.037 [-0.004, +0.082]; V6 - SelfCheckGPT: +0.111 [+0.048, +0.176]*; V6 - semantic_entropy: +0.269 [+0.203, +0.331]*; V6 - highconf_contradicted: +0.073 [+0.017, +0.129]*; V6 - V3: -0.007 [-0.017, +0.004]; ctx_lr+fb - ctx_lr: +0.005 [-0.025, +0.036]
- **mistral-small-24b**: V6 - V6-prior: +0.087 [+0.055, +0.121]*; V6 - ctx_lr: -0.005 [-0.049, +0.042]; V6 - ctx_lr+fb: -0.030 [-0.072, +0.012]; V6 - SelfCheckGPT: +0.135 [+0.058, +0.209]*; V6 - semantic_entropy: +0.315 [+0.248, +0.378]*; V6 - highconf_contradicted: +0.022 [-0.024, +0.068]; V6 - V3: -0.011 [-0.017, -0.005]*; ctx_lr+fb - ctx_lr: +0.026 [-0.007, +0.059]

Data efficiency (Amendment 1, descriptive): test AUROC vs labelled training sessions

| corpus | detector | 25 | 50 | 100 | 200 | all train |
|---|---|---|---|---|---|---|
| 8B | V6 | 0.765 | 0.780 | 0.802 | 0.814 | 0.829 |
| 8B | ctx_lr | 0.648 | 0.675 | 0.704 | 0.721 | 0.744 |
| 8B | ctx_lr+fb | 0.696 | 0.723 | 0.751 | 0.775 | 0.810 |
| qwen72b | V6 | 0.829 | 0.826 | 0.859 | 0.871 | 0.881 |
| qwen72b | ctx_lr | 0.756 | 0.799 | 0.826 | 0.839 | 0.853 |
| qwen72b | ctx_lr+fb | 0.765 | 0.807 | 0.830 | 0.857 | 0.879 |
| gpt-4o-mini | V6 | 0.772 | 0.799 | 0.815 | 0.846 | 0.849 |
| gpt-4o-mini | ctx_lr | 0.757 | 0.785 | 0.791 | 0.803 | 0.806 |
| gpt-4o-mini | ctx_lr+fb | 0.741 | 0.786 | 0.790 | 0.792 | 0.809 |
| mistral-small-24b | V6 | 0.774 | 0.809 | 0.832 | 0.840 | 0.845 |
| mistral-small-24b | ctx_lr | 0.729 | 0.753 | 0.796 | 0.827 | 0.840 |
| mistral-small-24b | ctx_lr+fb | 0.734 | 0.757 | 0.817 | 0.852 | 0.872 |

Pooled over 4 corpora (stratified cluster bootstrap):

| id | quantity | estimate [95% CI] | licensed claim |
|---|---|---|---|
| H1 | AUROC(SelfCheckGPT) | 0.676 [0.645, 0.707] | report the AUROC; do not say 'near chance' |
| H1 | AUROC(semantic_entropy) | 0.549 [0.526, 0.573] | sampling detectors are near chance on user-specific hallucination |
| H2 | V6 - V6-prior | +0.080 [+0.064, +0.096]* | the user's implicit reactions add detectable signal beyond population base rates |
| H3 | V6 - ctx_lr | +0.040 [+0.015, +0.063]* | PGE beats the strongest context-only detector |
| H4 | V6 - SelfCheckGPT | +0.175 [+0.135, +0.207]* | PGE beats sampling-consistency detectors |
| H4 | V6 - semantic_entropy | +0.302 [+0.269, +0.329]* | PGE beats sampling-consistency detectors |
| H2b | ctx_lr+fb - ctx_lr | +0.031 [+0.017, +0.047]* | the user's reactions add detectable signal beyond the commitments themselves (supervised detector) |
| H4b | ctx_lr+fb - SelfCheckGPT | +0.167 [+0.131, +0.199]* | detectors on PGE observables beat sampling-consistency detectors |
| H4b | ctx_lr+fb - semantic_entropy | +0.293 [+0.260, +0.322]* | detectors on PGE observables beat sampling-consistency detectors |
| sec. | V6 - ctx_lr+fb | +0.009 [-0.012, +0.027] | reported as is |
| sec. | V6 - V3 | -0.004 [-0.008, -0.001]* | reported as is |

## Per-commitment check of Corollary 1 (user-dependent commitments; descriptive)

| corpus | n | wrong | all resamples agree | e1 = P(disagree|wrong) | e0 = P(disagree|right) | Cor. 1 bound | AUROC disagreement | AUROC one feedback bit |
|---|---|---|---|---|---|---|---|---|
| 8B | 1310 | 0.397 | 0.495 | 0.544 | 0.478 | 0.881 | 0.555 | 0.746 |
| qwen72b | 1310 | 0.431 | 0.831 | 0.189 | 0.154 | 0.657 | 0.517 | 0.744 |
| gpt-4o-mini | 1310 | 0.429 | 0.856 | 0.183 | 0.115 | 0.639 | 0.535 | 0.746 |
| mistral-small-24b | 1310 | 0.406 | 0.692 | 0.344 | 0.283 | 0.765 | 0.534 | 0.746 |

## Amendment 2: rules-aware analyses on L_U (oracle information: the simulator's rules and hidden-fact prior; not deployable)

| corpus | eta0 (Bayes, no feedback) | etaX (Bayes, shown reactions) | etaF (Bayes, feedback) | V6-session | LR-session | LR-session+fb |
|---|---|---|---|---|---|---|
| 8B | 0.899 | 0.952 | 0.953 | 0.904 | 0.875 | 0.908 |
| qwen72b | 0.854 | 0.930 | 0.932 | 0.879 | 0.799 | 0.851 |
| gpt-4o-mini | 0.803 | 0.907 | 0.909 | 0.837 | 0.748 | 0.825 |
| mistral-small-24b | 0.781 | 0.890 | 0.891 | 0.847 | 0.669 | 0.814 |

Differences per corpus (session-clustered):

- **8B**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.054 [+0.028, +0.082]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): +0.000 [-0.000, +0.002]; V6-session - LR-session+fb: -0.004 [-0.030, +0.024]; V6-session - eta0 (Bayes, no feedback): +0.004 [-0.029, +0.035]; LR-session+fb - LR-session: +0.034 [-0.001, +0.068]
- **qwen72b**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.078 [+0.043, +0.114]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): +0.001 [-0.003, +0.006]; V6-session - LR-session+fb: +0.029 [-0.009, +0.068]; V6-session - eta0 (Bayes, no feedback): +0.023 [-0.022, +0.065]; LR-session+fb - LR-session: +0.052 [+0.012, +0.093]*
- **gpt-4o-mini**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.108 [+0.068, +0.152]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): +0.003 [-0.003, +0.008]; V6-session - LR-session+fb: +0.014 [-0.025, +0.054]; V6-session - eta0 (Bayes, no feedback): +0.039 [-0.012, +0.086]; LR-session+fb - LR-session: +0.085 [+0.032, +0.136]*
- **mistral-small-24b**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.111 [+0.068, +0.154]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): +0.001 [-0.007, +0.008]; V6-session - LR-session+fb: +0.032 [-0.011, +0.076]; V6-session - eta0 (Bayes, no feedback): +0.065 [+0.011, +0.118]*; LR-session+fb - LR-session: +0.148 [+0.094, +0.203]*

Pooled over 4 corpora:

| id | quantity | estimate [95% CI] | licensed claim |
|---|---|---|---|
| H5 | etaF (Bayes, feedback) - eta0 (Bayes, no feedback) | +0.087 [+0.069, +0.107]* | the user's reactions carry information beyond the population risk of the commitments |
| H6 | V6-session - LR-session+fb | +0.017 [+0.000, +0.036]* | given the same information, the PGE posterior beats the supervised detector |
| sec. | V6-session - eta0 (Bayes, no feedback) | +0.033 [+0.010, +0.056]* | reported as is |
| sec. | etaF (Bayes, feedback) - etaX (Bayes, shown reactions) | +0.001 [-0.001, +0.004] | reported as is |

runtime 908s
