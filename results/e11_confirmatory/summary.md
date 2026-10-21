# e11 confirmatory corpora (seed 7, protocol v2fb)

Implements results/PREREG_confirmatory.md. Fit on train, test once; split seeds 42/123/456.
L_U detectors: V6/V3 label event restricted to USER_DEPENDENT_SCHEMA and trained on L_U;
sampling baselines on user-dependent turns only. L detectors: as in e6.


## user-specific label L_U (primary)

Session AUROC, mean over split seeds 42/123/456:

| corpus | V6 | V3 | V6-prior | ctx_lr | ctx_lr+fb | SelfCheckGPT | semantic_entropy | highconf_contradicted | neg_signal_fraction | dag_logreg | gcn | mean_confidence (oriented) | best graphon | cove |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 70B | 0.695 | 0.686 | 0.634 | 0.671 | 0.745 | 0.492 | 0.497 | 0.613 | 0.563 | 0.631 | 0.575 | 0.550 | 0.483 | 0.510 |
| 8B | 0.672 | 0.669 | 0.591 | 0.686 | 0.725 | 0.610 | 0.533 | 0.614 | 0.536 | 0.647 | 0.574 | 0.512 | 0.531 |
| qwen72b | 0.650 | 0.654 | 0.593 | 0.715 | 0.748 | 0.559 | 0.485 | 0.628 | 0.579 | 0.653 | 0.663 | 0.522 | 0.506 |
| gpt-4o-mini | 0.636 | 0.635 | 0.575 | 0.635 | 0.668 | 0.576 | 0.486 | 0.576 | 0.595 | 0.609 | 0.597 | 0.465 | 0.521 |
| mistral-small-24b | 0.666 | 0.673 | 0.577 | 0.556 | 0.682 | 0.520 | 0.474 | 0.637 | 0.624 | 0.596 | 0.569 | 0.454 | 0.409 |

Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:

- **70B**: V6 - V6-prior: +0.060 [+0.032, +0.091]*; V6 - ctx_lr: +0.026 [-0.039, +0.092]; V6 - ctx_lr+fb: -0.043 [-0.097, +0.012]; V6 - SelfCheckGPT: +0.201 [+0.131, +0.275]*; V6 - semantic_entropy: +0.197 [+0.138, +0.254]*; V6 - highconf_contradicted: +0.082 [+0.023, +0.142]*; V6 - V3: +0.008 [+0.001, +0.015]*; ctx_lr+fb - ctx_lr: +0.070 [+0.024, +0.114]*; V6 - cove: +0.183 [+0.112, +0.253]*
- **8B**: V6 - V6-prior: +0.079 [+0.042, +0.116]*; V6 - ctx_lr: -0.016 [-0.082, +0.052]; V6 - ctx_lr+fb: -0.056 [-0.111, -0.001]*; V6 - SelfCheckGPT: +0.060 [-0.015, +0.134]; V6 - semantic_entropy: +0.139 [+0.067, +0.209]*; V6 - highconf_contradicted: +0.055 [-0.007, +0.117]; V6 - V3: +0.001 [-0.008, +0.011]; ctx_lr+fb - ctx_lr: +0.040 [+0.002, +0.079]*
- **qwen72b**: V6 - V6-prior: +0.057 [+0.023, +0.091]*; V6 - ctx_lr: -0.070 [-0.137, -0.002]*; V6 - ctx_lr+fb: -0.103 [-0.163, -0.044]*; V6 - SelfCheckGPT: +0.086 [+0.014, +0.162]*; V6 - semantic_entropy: +0.159 [+0.096, +0.220]*; V6 - highconf_contradicted: +0.019 [-0.027, +0.063]; V6 - V3: -0.007 [-0.016, +0.003]; ctx_lr+fb - ctx_lr: +0.033 [-0.009, +0.075]
- **gpt-4o-mini**: V6 - V6-prior: +0.060 [+0.026, +0.095]*; V6 - ctx_lr: +0.005 [-0.070, +0.079]; V6 - ctx_lr+fb: -0.029 [-0.099, +0.040]; V6 - SelfCheckGPT: +0.058 [+0.004, +0.111]*; V6 - semantic_entropy: +0.151 [+0.090, +0.215]*; V6 - highconf_contradicted: +0.058 [-0.002, +0.114]; V6 - V3: -0.000 [-0.008, +0.007]; ctx_lr+fb - ctx_lr: +0.034 [-0.004, +0.075]
- **mistral-small-24b**: V6 - V6-prior: +0.090 [+0.052, +0.127]*; V6 - ctx_lr: +0.115 [+0.042, +0.187]*; V6 - ctx_lr+fb: -0.011 [-0.073, +0.051]; V6 - SelfCheckGPT: +0.150 [+0.079, +0.222]*; V6 - semantic_entropy: +0.197 [+0.138, +0.255]*; V6 - highconf_contradicted: +0.032 [-0.019, +0.083]; V6 - V3: -0.005 [-0.011, +0.001]; ctx_lr+fb - ctx_lr: +0.126 [+0.077, +0.175]*

Data efficiency (Amendment 1, descriptive): test AUROC vs labelled training sessions

| corpus | detector | 25 | 50 | 100 | 200 | all train |
|---|---|---|---|---|---|---|
| 70B | V6 | 0.650 | 0.666 | 0.678 | 0.685 | 0.695 |
| 70B | ctx_lr | 0.573 | 0.601 | 0.628 | 0.646 | 0.671 |
| 70B | ctx_lr+fb | 0.593 | 0.642 | 0.679 | 0.707 | 0.745 |
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

Pooled over 5 corpora (stratified cluster bootstrap):

| id | quantity | estimate [95% CI] | licensed claim |
|---|---|---|---|
| H1 | AUROC(SelfCheckGPT) | 0.551 [0.524, 0.578] | sampling detectors are near chance on user-specific hallucination |
| H1 | AUROC(semantic_entropy) | 0.495 [0.481, 0.508] | sampling detectors are near chance on user-specific hallucination |
| H2 | V6 - V6-prior | +0.070 [+0.055, +0.086]* | the user's implicit reactions add detectable signal beyond population base rates |
| H3 | V6 - ctx_lr | +0.011 [-0.021, +0.043] | context alone matches PGE on this label |
| H4 | V6 - SelfCheckGPT | +0.112 [+0.081, +0.143]* | PGE beats sampling-consistency detectors |
| H4 | V6 - semantic_entropy | +0.169 [+0.139, +0.198]* | PGE beats sampling-consistency detectors |
| H2b | ctx_lr+fb - ctx_lr | +0.061 [+0.041, +0.080]* | the user's reactions add detectable signal beyond the commitments themselves (supervised detector) |
| H4b | ctx_lr+fb - SelfCheckGPT | +0.162 [+0.126, +0.195]* | detectors on PGE observables beat sampling-consistency detectors |
| H4b | ctx_lr+fb - semantic_entropy | +0.219 [+0.190, +0.243]* | detectors on PGE observables beat sampling-consistency detectors |
| sec. | V6 - ctx_lr+fb | -0.050 [-0.076, -0.021]* | reported as is |
| sec. | V6 - V3 | +0.000 [-0.004, +0.003] | reported as is |

## composite label L (secondary)

Session AUROC, mean over split seeds 42/123/456:

| corpus | V6 | V3 | V6-prior | ctx_lr | ctx_lr+fb | SelfCheckGPT | semantic_entropy | highconf_contradicted | neg_signal_fraction | dag_logreg | gcn | mean_confidence (oriented) | best graphon | cove |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 70B | 0.863 | 0.852 | 0.820 | 0.817 | 0.832 | 0.520 | 0.539 | 0.763 | 0.702 | 0.761 | 0.700 | 0.700 | 0.460 | 0.603 |
| 8B | 0.829 | 0.828 | 0.717 | 0.744 | 0.810 | 0.641 | 0.572 | 0.785 | 0.713 | 0.783 | 0.708 | 0.642 | 0.533 |
| qwen72b | 0.881 | 0.883 | 0.796 | 0.853 | 0.879 | 0.634 | 0.530 | 0.846 | 0.826 | 0.863 | 0.812 | 0.718 | 0.448 |
| gpt-4o-mini | 0.849 | 0.857 | 0.815 | 0.806 | 0.809 | 0.730 | 0.571 | 0.770 | 0.761 | 0.810 | 0.771 | 0.756 | 0.419 |
| mistral-small-24b | 0.845 | 0.852 | 0.754 | 0.840 | 0.872 | 0.698 | 0.524 | 0.821 | 0.812 | 0.821 | 0.764 | 0.679 | 0.414 |

Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:

- **70B**: V6 - V6-prior: +0.042 [+0.016, +0.070]*; V6 - ctx_lr: +0.044 [+0.012, +0.076]*; V6 - ctx_lr+fb: +0.030 [-0.005, +0.068]; V6 - SelfCheckGPT: +0.340 [+0.268, +0.409]*; V6 - semantic_entropy: +0.321 [+0.269, +0.368]*; V6 - highconf_contradicted: +0.096 [+0.048, +0.143]*; V6 - V3: +0.004 [-0.007, +0.014]; ctx_lr+fb - ctx_lr: +0.013 [-0.009, +0.036]; V6 - cove: +0.257 [+0.190, +0.328]*
- **8B**: V6 - V6-prior: +0.113 [+0.074, +0.152]*; V6 - ctx_lr: +0.086 [+0.028, +0.145]*; V6 - ctx_lr+fb: +0.018 [-0.022, +0.058]; V6 - SelfCheckGPT: +0.186 [+0.115, +0.257]*; V6 - semantic_entropy: +0.255 [+0.187, +0.322]*; V6 - highconf_contradicted: +0.044 [+0.009, +0.080]*; V6 - V3: +0.002 [-0.004, +0.007]; ctx_lr+fb - ctx_lr: +0.067 [+0.037, +0.099]*
- **qwen72b**: V6 - V6-prior: +0.085 [+0.054, +0.117]*; V6 - ctx_lr: +0.031 [-0.002, +0.067]; V6 - ctx_lr+fb: +0.003 [-0.026, +0.033]; V6 - SelfCheckGPT: +0.250 [+0.181, +0.320]*; V6 - semantic_entropy: +0.352 [+0.301, +0.399]*; V6 - highconf_contradicted: +0.035 [+0.001, +0.069]*; V6 - V3: -0.003 [-0.007, +0.001]; ctx_lr+fb - ctx_lr: +0.028 [+0.004, +0.054]*
- **gpt-4o-mini**: V6 - V6-prior: +0.037 [+0.014, +0.062]*; V6 - ctx_lr: +0.042 [-0.004, +0.092]; V6 - ctx_lr+fb: +0.037 [-0.004, +0.082]; V6 - SelfCheckGPT: +0.111 [+0.048, +0.176]*; V6 - semantic_entropy: +0.269 [+0.203, +0.331]*; V6 - highconf_contradicted: +0.073 [+0.017, +0.129]*; V6 - V3: -0.007 [-0.017, +0.004]; ctx_lr+fb - ctx_lr: +0.005 [-0.025, +0.036]
- **mistral-small-24b**: V6 - V6-prior: +0.087 [+0.055, +0.121]*; V6 - ctx_lr: -0.005 [-0.049, +0.042]; V6 - ctx_lr+fb: -0.030 [-0.072, +0.012]; V6 - SelfCheckGPT: +0.135 [+0.058, +0.209]*; V6 - semantic_entropy: +0.315 [+0.248, +0.378]*; V6 - highconf_contradicted: +0.022 [-0.024, +0.068]; V6 - V3: -0.011 [-0.017, -0.005]*; ctx_lr+fb - ctx_lr: +0.026 [-0.007, +0.059]

Data efficiency (Amendment 1, descriptive): test AUROC vs labelled training sessions

| corpus | detector | 25 | 50 | 100 | 200 | all train |
|---|---|---|---|---|---|---|
| 70B | V6 | 0.786 | 0.817 | 0.839 | 0.853 | 0.863 |
| 70B | ctx_lr | 0.734 | 0.771 | 0.777 | 0.803 | 0.817 |
| 70B | ctx_lr+fb | 0.736 | 0.773 | 0.789 | 0.809 | 0.832 |
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

Pooled over 5 corpora (stratified cluster bootstrap):

| id | quantity | estimate [95% CI] | licensed claim |
|---|---|---|---|
| H1 | AUROC(SelfCheckGPT) | 0.645 [0.617, 0.672] | report the AUROC; do not say 'near chance' |
| H1 | AUROC(semantic_entropy) | 0.547 [0.527, 0.569] | sampling detectors are near chance on user-specific hallucination |
| H2 | V6 - V6-prior | +0.073 [+0.059, +0.087]* | the user's implicit reactions add detectable signal beyond population base rates |
| H3 | V6 - ctx_lr | +0.041 [+0.021, +0.060]* | PGE beats the strongest context-only detector |
| H4 | V6 - SelfCheckGPT | +0.209 [+0.173, +0.237]* | PGE beats sampling-consistency detectors |
| H4 | V6 - semantic_entropy | +0.306 [+0.276, +0.328]* | PGE beats sampling-consistency detectors |
| H2b | ctx_lr+fb - ctx_lr | +0.028 [+0.016, +0.041]* | the user's reactions add detectable signal beyond the commitments themselves (supervised detector) |
| H4b | ctx_lr+fb - SelfCheckGPT | +0.196 [+0.163, +0.223]* | detectors on PGE observables beat sampling-consistency detectors |
| H4b | ctx_lr+fb - semantic_entropy | +0.293 [+0.263, +0.318]* | detectors on PGE observables beat sampling-consistency detectors |
| sec. | V6 - ctx_lr+fb | +0.013 [-0.005, +0.029] | reported as is |
| sec. | V6 - V3 | -0.001 [-0.006, +0.000] | reported as is |

## Per-commitment check of Corollary 1 (user-dependent commitments; descriptive)

| corpus | n | wrong | all resamples agree | e1 = P(disagree|wrong) | e0 = P(disagree|right) | Cor. 1 bound | AUROC disagreement | AUROC one feedback bit |
|---|---|---|---|---|---|---|---|---|
| 70B | 1310 | 0.524 | 0.914 | 0.087 | 0.085 | 0.582 | 0.501 | 0.748 |
| 8B | 1310 | 0.397 | 0.495 | 0.544 | 0.478 | 0.881 | 0.555 | 0.746 |
| qwen72b | 1310 | 0.431 | 0.831 | 0.189 | 0.154 | 0.657 | 0.517 | 0.744 |
| gpt-4o-mini | 1310 | 0.429 | 0.856 | 0.183 | 0.115 | 0.639 | 0.535 | 0.746 |
| mistral-small-24b | 1310 | 0.406 | 0.692 | 0.344 | 0.283 | 0.765 | 0.534 | 0.746 |

## Amendment 2: rules-aware analyses on L_U (oracle information: the simulator's rules and hidden-fact prior; not deployable)

| corpus | eta0 (Bayes, no feedback) | etaX (Bayes, shown reactions) | etaF (Bayes, feedback) | V6-session | LR-session | LR-session+fb |
|---|---|---|---|---|---|---|
| 70B | 0.876 | 0.942 | 0.944 | 0.913 | 0.850 | 0.885 |
| 8B | 0.899 | 0.952 | 0.953 | 0.904 | 0.875 | 0.908 |
| qwen72b | 0.854 | 0.930 | 0.932 | 0.879 | 0.799 | 0.851 |
| gpt-4o-mini | 0.803 | 0.907 | 0.909 | 0.837 | 0.748 | 0.825 |
| mistral-small-24b | 0.781 | 0.890 | 0.891 | 0.847 | 0.669 | 0.814 |

Differences per corpus (session-clustered):

- **70B**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.067 [+0.041, +0.097]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): +0.002 [-0.002, +0.006]; V6-session - LR-session+fb: +0.032 [+0.007, +0.058]*; V6-session - eta0 (Bayes, no feedback): +0.035 [+0.003, +0.069]*; LR-session+fb - LR-session: +0.032 [+0.001, +0.065]*
- **8B**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.054 [+0.028, +0.082]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): +0.000 [-0.000, +0.002]; V6-session - LR-session+fb: -0.004 [-0.030, +0.024]; V6-session - eta0 (Bayes, no feedback): +0.004 [-0.029, +0.035]; LR-session+fb - LR-session: +0.034 [-0.001, +0.068]
- **qwen72b**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.078 [+0.043, +0.114]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): +0.001 [-0.003, +0.006]; V6-session - LR-session+fb: +0.029 [-0.009, +0.068]; V6-session - eta0 (Bayes, no feedback): +0.023 [-0.022, +0.065]; LR-session+fb - LR-session: +0.052 [+0.012, +0.093]*
- **gpt-4o-mini**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.108 [+0.068, +0.152]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): +0.003 [-0.003, +0.008]; V6-session - LR-session+fb: +0.014 [-0.025, +0.054]; V6-session - eta0 (Bayes, no feedback): +0.039 [-0.012, +0.086]; LR-session+fb - LR-session: +0.085 [+0.032, +0.136]*
- **mistral-small-24b**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.111 [+0.068, +0.154]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): +0.001 [-0.007, +0.008]; V6-session - LR-session+fb: +0.032 [-0.011, +0.076]; V6-session - eta0 (Bayes, no feedback): +0.065 [+0.011, +0.118]*; LR-session+fb - LR-session: +0.148 [+0.094, +0.203]*

Pooled over 5 corpora:

| id | quantity | estimate [95% CI] | licensed claim |
|---|---|---|---|
| H5 | etaF (Bayes, feedback) - eta0 (Bayes, no feedback) | +0.083 [+0.068, +0.100]* | the user's reactions carry information beyond the population risk of the commitments |
| H6 | V6-session - LR-session+fb | +0.019 [+0.006, +0.037]* | given the same information, the PGE posterior beats the supervised detector |
| sec. | V6-session - eta0 (Bayes, no feedback) | +0.033 [+0.013, +0.053]* | reported as is |
| sec. | etaF (Bayes, feedback) - etaX (Bayes, shown reactions) | +0.001 [-0.001, +0.004] | reported as is |

# LLM-simulated patient (behavioral feedback)

> 70B LLM-patient (profile): `data/trajectories/healthcare_llmpatient/70B_7.jsonl` not found; skipped.
> 70B LLM-patient (informed): `data/trajectories/healthcare_llmpatient/70B_7_informed.jsonl` not found; skipped.

## user-specific label L_U (primary)

Session AUROC, mean over split seeds 42/123/456:

| corpus | V6 | V3 | V6-prior | ctx_lr | ctx_lr+fb | highconf_contradicted | neg_signal_fraction | dag_logreg | mean_confidence (oriented) |
|---|---|---|---|---|---|---|---|---|---|
| 70B LLM-patient v1 pilot (profile) | 0.756 | 0.675 | 0.749 | 0.773 | 0.739 | 0.541 | 0.534 | 0.595 | 0.614 |
| 70B LLM-patient v1 pilot (informed) | 0.864 | 0.881 | 0.840 | 0.693 | 0.743 | 0.554 | 0.554 | 0.670 | 0.581 |
| 70B LLM-patient v2 (profile) | 0.644 | 0.664 | 0.601 | 0.729 | 0.733 | 0.581 | 0.551 | 0.653 | 0.601 |
| 70B LLM-patient v2 pilot (profile) | 0.582 | 0.594 | 0.525 | 0.706 | 0.725 | 0.579 | 0.544 | 0.512 | 0.668 |

Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:

- **70B LLM-patient v1 pilot (profile)**: V6 - V6-prior: +0.017 [-0.048, +0.084]; V6 - ctx_lr: -0.021 [-0.176, +0.119]; V6 - ctx_lr+fb: +0.017 [-0.121, +0.158]; V6 - highconf_contradicted: +0.216 [+0.036, +0.374]*; V6 - V3: +0.097 [-0.031, +0.237]; ctx_lr+fb - ctx_lr: -0.037 [-0.091, +0.006]
- **70B LLM-patient v1 pilot (informed)**: V6 - V6-prior: +0.032 [-0.007, +0.077]; V6 - ctx_lr: +0.160 [+0.048, +0.279]*; V6 - ctx_lr+fb: +0.128 [+0.031, +0.226]*; V6 - highconf_contradicted: +0.299 [+0.186, +0.407]*; V6 - V3: -0.015 [-0.071, +0.043]; ctx_lr+fb - ctx_lr: +0.032 [-0.035, +0.104]
- **70B LLM-patient v2 (profile)**: V6 - V6-prior: +0.051 [+0.023, +0.076]*; V6 - ctx_lr: -0.081 [-0.141, -0.026]*; V6 - ctx_lr+fb: -0.086 [-0.147, -0.025]*; V6 - highconf_contradicted: +0.067 [+0.008, +0.130]*; V6 - V3: -0.020 [-0.051, +0.010]; ctx_lr+fb - ctx_lr: +0.005 [-0.034, +0.043]
- **70B LLM-patient v2 pilot (profile)**: V6 - V6-prior: +0.066 [-0.034, +0.170]; V6 - ctx_lr: -0.090 [-0.258, +0.079]; V6 - ctx_lr+fb: -0.102 [-0.262, +0.070]; V6 - highconf_contradicted: -0.006 [-0.130, +0.122]; V6 - V3: -0.003 [-0.077, +0.076]; ctx_lr+fb - ctx_lr: +0.012 [-0.082, +0.101]

## composite label L (secondary)

Session AUROC, mean over split seeds 42/123/456:

| corpus | V6 | V3 | V6-prior | ctx_lr | ctx_lr+fb | highconf_contradicted | neg_signal_fraction | dag_logreg | mean_confidence (oriented) |
|---|---|---|---|---|---|---|---|---|---|
| 70B LLM-patient v1 pilot (profile) | 0.826 | 0.708 | 0.882 | 0.816 | 0.781 | 0.538 | 0.561 | 0.588 | 0.598 |
| 70B LLM-patient v1 pilot (informed) | 0.785 | 0.654 | 0.750 | 0.760 | 0.773 | 0.542 | 0.555 | 0.572 | 0.544 |
| 70B LLM-patient v2 (profile) | 0.743 | 0.764 | 0.754 | 0.769 | 0.731 | 0.623 | 0.586 | 0.673 | 0.630 |
| 70B LLM-patient v2 pilot (profile) | 0.760 | 0.739 | 0.729 | 0.717 | 0.688 | 0.685 | 0.603 | 0.535 | 0.560 |

Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:

- **70B LLM-patient v1 pilot (profile)**: V6 - V6-prior: -0.049 [-0.120, +0.014]; V6 - ctx_lr: +0.029 [-0.076, +0.144]; V6 - ctx_lr+fb: +0.049 [-0.069, +0.175]; V6 - highconf_contradicted: +0.294 [+0.195, +0.383]*; V6 - V3: +0.130 [+0.004, +0.269]*; ctx_lr+fb - ctx_lr: -0.020 [-0.076, +0.024]
- **70B LLM-patient v1 pilot (informed)**: V6 - V6-prior: +0.040 [-0.059, +0.129]; V6 - ctx_lr: +0.054 [-0.135, +0.268]; V6 - ctx_lr+fb: +0.039 [-0.162, +0.256]; V6 - highconf_contradicted: +0.238 [+0.106, +0.351]*; V6 - V3: +0.148 [-0.000, +0.292]; ctx_lr+fb - ctx_lr: +0.016 [-0.028, +0.060]
- **70B LLM-patient v2 (profile)**: V6 - V6-prior: -0.013 [-0.045, +0.017]; V6 - ctx_lr: -0.028 [-0.075, +0.019]; V6 - ctx_lr+fb: +0.011 [-0.045, +0.065]; V6 - highconf_contradicted: +0.118 [+0.051, +0.182]*; V6 - V3: -0.021 [-0.044, -0.002]*; ctx_lr+fb - ctx_lr: -0.038 [-0.067, -0.011]*
- **70B LLM-patient v2 pilot (profile)**: V6 - V6-prior: +0.077 [-0.013, +0.182]; V6 - ctx_lr: +0.060 [-0.051, +0.172]; V6 - ctx_lr+fb: +0.091 [-0.034, +0.210]; V6 - highconf_contradicted: +0.090 [-0.043, +0.228]; V6 - V3: -0.002 [-0.058, +0.053]; ctx_lr+fb - ctx_lr: -0.031 [-0.097, +0.035]

## Amendment 2: rules-aware analyses on L_U (oracle information: the simulator's rules and hidden-fact prior; not deployable)

| corpus | eta0 (Bayes, no feedback) | etaX (Bayes, shown reactions) | etaF (Bayes, feedback) | V6-session | LR-session | LR-session+fb |
|---|---|---|---|---|---|---|
| 70B LLM-patient v1 pilot (profile) | 0.935 | n/a | n/a | 0.937 | 0.957 | 0.959 |
| 70B LLM-patient v1 pilot (informed) | 0.936 | n/a | n/a | 0.954 | 0.893 | 0.889 |
| 70B LLM-patient v2 (profile) | 0.854 | n/a | n/a | 0.793 | 0.818 | 0.826 |
| 70B LLM-patient v2 pilot (profile) | 0.868 | n/a | n/a | 0.726 | 0.853 | 0.851 |

Differences per corpus (session-clustered):

- **70B LLM-patient v1 pilot (profile)**: V6-session - LR-session+fb: -0.025 [-0.085, +0.024]; V6-session - eta0 (Bayes, no feedback): +0.007 [-0.042, +0.059]; LR-session+fb - LR-session: +0.006 [-0.003, +0.023]
- **70B LLM-patient v1 pilot (informed)**: V6-session - LR-session+fb: +0.077 [+0.016, +0.151]*; V6-session - eta0 (Bayes, no feedback): +0.024 [-0.007, +0.063]; LR-session+fb - LR-session: +0.002 [-0.011, +0.017]
- **70B LLM-patient v2 (profile)**: V6-session - LR-session+fb: -0.036 [-0.086, +0.016]; V6-session - eta0 (Bayes, no feedback): -0.065 [-0.105, -0.025]*; LR-session+fb - LR-session: +0.008 [-0.022, +0.041]
- **70B LLM-patient v2 pilot (profile)**: V6-session - LR-session+fb: -0.082 [-0.214, +0.049]; V6-session - eta0 (Bayes, no feedback): -0.134 [-0.233, -0.051]*; LR-session+fb - LR-session: -0.000 [-0.051, +0.052]

## Measured clarification: pre-registered comparisons (paired bootstrap over sessions)

120 sessions. Primary outcome: wrong confident user-dependent commitments per session.

| condition | questions | L_U rate | wrong conf. user-dep. | L rate | wrong conf. |
|---|---|---|---|---|---|
| cached | 0.00 | 0.467 | 0.642 | 0.350 | 1.850 |
| detailed_questions | 4.29 | 0.042 | 0.108 | 0.042 | 1.625 |
| rerun_full | 0.00 | 0.483 | 0.617 | 0.275 | 1.883 |
| posterior@1 | 1.00 | 0.383 | 0.575 | 0.142 | 1.683 |
| no_question@1 | 0.00 | 0.492 | 0.700 | 0.300 | 1.975 |
| posterior@2 | 2.00 | 0.358 | 0.542 | 0.025 | 1.417 |
| no_question@2 | 0.00 | 0.475 | 0.667 | 0.275 | 1.917 |
| random@1 | 1.00 | 0.392 | 0.583 | 0.167 | 1.692 |
| lowconf@1 | 1.00 | 0.458 | 0.650 | 0.150 | 1.833 |

| A - B | wrong conf. user-dep. (primary) | L_U | L |
|---|---|---|---|
| posterior@1 - no_question@1 | -0.125 [-0.242, +0.000] | -0.108 [-0.200, -0.016]* | -0.158 [-0.242, -0.083]* |
| posterior@1 - random@1 | -0.008 [-0.133, +0.117] | -0.008 [-0.100, +0.083] | -0.025 [-0.083, +0.033] |
| posterior@1 - lowconf@1 | -0.075 [-0.183, +0.025] | -0.075 [-0.150, +0.000] | -0.008 [-0.075, +0.058] |
| posterior@2 - detailed_questions | +0.433 [+0.325, +0.550]* | +0.317 [+0.233, +0.400]* | -0.017 [-0.050, +0.017] |
| posterior@2 - rerun_full | -0.075 [-0.208, +0.058] | -0.125 [-0.217, -0.033]* | -0.250 [-0.333, -0.175]* |
| detailed_questions - rerun_full | -0.508 [-0.650, -0.375]* | -0.442 [-0.533, -0.350]* | -0.233 [-0.317, -0.158]* |
| posterior@1 - detailed_questions | +0.467 [+0.358, +0.584]* | +0.342 [+0.258, +0.425]* | +0.100 [+0.042, +0.158]* |
| posterior@1 - cached | -0.067 [-0.200, +0.058] | -0.083 [-0.183, +0.025] | -0.208 [-0.292, -0.125]* |
| rerun_full - cached | -0.025 [-0.142, +0.083] | +0.017 [-0.058, +0.092] | -0.075 [-0.158, +0.008] |

# Measured closed-loop clarification (results/e8_closed_loop/70B_7/summary.txt)

```
# closed-loop clarification (paid, measured) — meta-llama/Llama-3.3-70B-Instruct-Turbo, cache data/trajectories/healthcare_confirm/70B_7.jsonl, test split seed 42 (120 sessions), protocol v2fb, picker v6, noise stream (seed, 7919, i), cost $3.741
## outcome: composite label (eval_label_v2)
paired sessions with every condition complete: 120

condition            questions/sess hall. rate wrong conf/sess rerun turns     cost
cached                         0.00      0.350           1.850           0    0.000
detailed_questions             4.29      0.042           1.625        1198    0.653
rerun_full                     0.00      0.275           1.883        1198    0.532
posterior@1                    1.00      0.142           1.683         777    0.385
no_question@1                  0.00      0.300           1.975         777    0.359
posterior@2                    2.00      0.025           1.417         933    0.464
no_question@2                  0.00      0.275           1.917         933    0.425
random@1                       1.00      0.167           1.692         760    0.376
lowconf@1                      1.00      0.150           1.833        1125    0.541

## paired bootstrap over sessions (2000): A - B, negative = A better
A                    B                                    d hall. rate            d wrong conf/sess
posterior@1          no_question@1        -0.158 [-0.242, -0.083]     -0.292 [-0.467, -0.117]
posterior@1          detailed_questions   +0.100 [+0.042, +0.158]     +0.058 [-0.158, +0.267]
posterior@1          cached               -0.208 [-0.292, -0.125]     -0.167 [-0.358, +0.017]
posterior@2          no_question@2        -0.250 [-0.333, -0.175]     -0.500 [-0.700, -0.300]
posterior@2          detailed_questions   -0.017 [-0.050, +0.017]     -0.208 [-0.408, -0.017]
posterior@2          cached               -0.325 [-0.408, -0.242]     -0.433 [-0.625, -0.233]
detailed_questions   cached               -0.308 [-0.400, -0.225]     -0.225 [-0.433, -0.017]
no_question@1        cached               -0.050 [-0.125, +0.025]     +0.125 [-0.042, +0.283]
no_question@2        cached               -0.075 [-0.158, +0.000]     +0.067 [-0.100, +0.242]
detailed_questions   rerun_full           -0.233 [-0.317, -0.158]     -0.258 [-0.483, -0.033]
rerun_full           cached               -0.075 [-0.158, +0.008]     +0.033 [-0.167, +0.225]
posterior@1          random@1             -0.025 [-0.083, +0.033]     -0.008 [-0.192, +0.167]
random@1             cached               -0.183 [-0.258, -0.108]     -0.158 [-0.333, +0.017]
posterior@1          lowconf@1            -0.008 [-0.075, +0.058]     -0.150 [-0.367, +0.050]
lowconf@1            cached               -0.200 [-0.292, -0.108]     -0.017 [-0.217, +0.183]

## outcome: user-specific label L_U (eval_label_user)
paired sessions with every condition complete: 120

condition            questions/sess hall. rate wrong conf/sess rerun turns     cost
cached                         0.00      0.467           0.642           0    0.000
detailed_questions             4.29      0.042           0.108        1198    0.653
rerun_full                     0.00      0.483           0.617        1198    0.532
posterior@1                    1.00      0.383           0.575         777    0.385
no_question@1                  0.00      0.492           0.700         777    0.359
posterior@2                    2.00      0.358           0.542         933    0.464
no_question@2                  0.00      0.475           0.667         933    0.425
random@1                       1.00      0.392           0.583         760    0.376
lowconf@1                      1.00      0.458           0.650        1125    0.541

## paired bootstrap over sessions (2000): A - B, negative = A better
A                    B                                    d hall. rate            d wrong conf/sess
posterior@1          no_question@1        -0.108 [-0.200, -0.016]     -0.125 [-0.242, +0.000]
posterior@1          detailed_questions   +0.342 [+0.258, +0.425]     +0.467 [+0.358, +0.584]
posterior@1          cached               -0.083 [-0.183, +0.025]     -0.067 [-0.200, +0.058]
posterior@2          no_question@2        -0.117 [-0.217, -0.017]     -0.125 [-0.267, +0.025]
posterior@2          detailed_questions   +0.317 [+0.233, +0.400]     +0.433 [+0.325, +0.550]
posterior@2          cached               -0.108 [-0.209, +0.008]     -0.100 [-0.258, +0.058]
detailed_questions   cached               -0.425 [-0.517, -0.333]     -0.533 [-0.675, -0.392]
no_question@1        cached               +0.025 [-0.042, +0.092]     +0.058 [-0.033, +0.150]
no_question@2        cached               +0.008 [-0.067, +0.075]     +0.025 [-0.075, +0.117]
detailed_questions   rerun_full           -0.442 [-0.533, -0.350]     -0.508 [-0.650, -0.375]
rerun_full           cached               +0.017 [-0.058, +0.092]     -0.025 [-0.142, +0.083]
posterior@1          random@1             -0.008 [-0.100, +0.083]     -0.008 [-0.133, +0.117]
random@1             cached               -0.075 [-0.150, +0.008]     -0.058 [-0.167, +0.050]
posterior@1          lowconf@1            -0.075 [-0.150, +0.000]     -0.075 [-0.183, +0.025]
lowconf@1            cached               -0.008 [-0.117, +0.092]     +0.008 [-0.125, +0.158]
```

# Second domain: travel booking (travel_v1, protocol v2fb, seed 7; Amendment 4)

Same labels, detectors, splits, statistics and wording rules as the healthcare corpora, applied within this domain; L_U uses travel_v1.USER_DEPENDENT_SCHEMA; pooled rows pool the travel corpora only.


## user-specific label L_U (primary)

Session AUROC, mean over split seeds 42/123/456:

| corpus | V6 | V3 | V6-prior | ctx_lr | ctx_lr+fb | SelfCheckGPT | semantic_entropy | highconf_contradicted | neg_signal_fraction | dag_logreg | gcn | mean_confidence (oriented) | best graphon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini (travel) | 0.682 | 0.683 | 0.607 | 0.607 | 0.658 | 0.535 | 0.502 | 0.629 | 0.596 | 0.611 | 0.659 | 0.537 | 0.476 |
| 8B (travel) | 0.796 | 0.794 | 0.713 | 0.786 | 0.825 | 0.628 | 0.513 | 0.620 | 0.586 | 0.638 | 0.544 | 0.549 | 0.497 |

Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:

- **gpt-4o-mini (travel)**: V6 - V6-prior: +0.082 [+0.026, +0.138]*; V6 - ctx_lr: +0.084 [+0.003, +0.163]*; V6 - ctx_lr+fb: +0.028 [-0.034, +0.088]; V6 - SelfCheckGPT: +0.153 [+0.069, +0.232]*; V6 - semantic_entropy: +0.182 [+0.121, +0.241]*; V6 - highconf_contradicted: +0.056 [-0.012, +0.129]; V6 - V3: +0.002 [-0.003, +0.006]; ctx_lr+fb - ctx_lr: +0.056 [-0.006, +0.120]
- **8B (travel)**: V6 - V6-prior: +0.084 [+0.041, +0.126]*; V6 - ctx_lr: +0.012 [-0.053, +0.076]; V6 - ctx_lr+fb: -0.027 [-0.083, +0.034]; V6 - SelfCheckGPT: +0.171 [+0.091, +0.258]*; V6 - semantic_entropy: +0.284 [+0.229, +0.336]*; V6 - highconf_contradicted: +0.177 [+0.096, +0.257]*; V6 - V3: +0.003 [-0.005, +0.011]; ctx_lr+fb - ctx_lr: +0.039 [-0.002, +0.078]

Data efficiency (Amendment 1, descriptive): test AUROC vs labelled training sessions

| corpus | detector | 25 | 50 | 100 | 200 | all train |
|---|---|---|---|---|---|---|
| gpt-4o-mini (travel) | V6 | 0.632 | 0.647 | 0.662 | 0.673 | 0.682 |
| gpt-4o-mini (travel) | ctx_lr | 0.577 | 0.529 | 0.548 | 0.586 | 0.607 |
| gpt-4o-mini (travel) | ctx_lr+fb | 0.586 | 0.563 | 0.596 | 0.626 | 0.658 |
| 8B (travel) | V6 | 0.751 | 0.754 | 0.767 | 0.782 | 0.796 |
| 8B (travel) | ctx_lr | 0.689 | 0.717 | 0.745 | 0.769 | 0.786 |
| 8B (travel) | ctx_lr+fb | 0.705 | 0.721 | 0.772 | 0.788 | 0.825 |

Pooled over 2 corpora (stratified cluster bootstrap):

| id | quantity | estimate [95% CI] | licensed claim |
|---|---|---|---|
| H1 | AUROC(SelfCheckGPT) | 0.582 [0.533, 0.626] | report the AUROC; do not say 'near chance' |
| H1 | AUROC(semantic_entropy) | 0.507 [0.499, 0.520] | sampling detectors are near chance on user-specific hallucination |
| H2 | V6 - V6-prior | +0.079 [+0.049, +0.118]* | the user's implicit reactions add detectable signal beyond population base rates |
| H3 | V6 - ctx_lr | +0.043 [-0.005, +0.098] | context alone matches PGE on this label |
| H4 | V6 - SelfCheckGPT | +0.157 [+0.102, +0.221]* | PGE beats sampling-consistency detectors |
| H4 | V6 - semantic_entropy | +0.232 [+0.194, +0.272]* | PGE beats sampling-consistency detectors |
| H2b | ctx_lr+fb - ctx_lr | +0.045 [+0.010, +0.083]* | the user's reactions add detectable signal beyond the commitments themselves (supervised detector) |
| H4b | ctx_lr+fb - SelfCheckGPT | +0.159 [+0.101, +0.219]* | detectors on PGE observables beat sampling-consistency detectors |
| H4b | ctx_lr+fb - semantic_entropy | +0.234 [+0.186, +0.276]* | detectors on PGE observables beat sampling-consistency detectors |
| sec. | V6 - ctx_lr+fb | -0.002 [-0.041, +0.042] | reported as is |
| sec. | V6 - V3 | +0.001 [-0.002, +0.007] | reported as is |

## composite label L (secondary)

Session AUROC, mean over split seeds 42/123/456:

| corpus | V6 | V3 | V6-prior | ctx_lr | ctx_lr+fb | SelfCheckGPT | semantic_entropy | highconf_contradicted | neg_signal_fraction | dag_logreg | gcn | mean_confidence (oriented) | best graphon |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini (travel) | 0.876 | 0.882 | 0.801 | 0.766 | 0.834 | 0.657 | 0.483 | 0.792 | 0.735 | 0.854 | 0.847 | 0.741 | 0.491 |
| 8B (travel) | 0.793 | 0.794 | 0.713 | 0.733 | 0.776 | 0.695 | 0.616 | 0.751 | 0.747 | 0.734 | 0.629 | 0.556 | 0.539 |

Differences per corpus, session-clustered paired bootstrap (2000); * = CI excludes 0:

- **gpt-4o-mini (travel)**: V6 - V6-prior: +0.074 [+0.034, +0.119]*; V6 - ctx_lr: +0.110 [+0.037, +0.183]*; V6 - ctx_lr+fb: +0.045 [-0.004, +0.097]; V6 - SelfCheckGPT: +0.216 [+0.139, +0.295]*; V6 - semantic_entropy: +0.388 [+0.317, +0.455]*; V6 - highconf_contradicted: +0.085 [+0.031, +0.140]*; V6 - V3: -0.009 [-0.018, -0.000]*; ctx_lr+fb - ctx_lr: +0.065 [+0.012, +0.120]*
- **8B (travel)**: V6 - V6-prior: +0.083 [+0.035, +0.130]*; V6 - ctx_lr: +0.068 [+0.011, +0.124]*; V6 - ctx_lr+fb: +0.021 [-0.022, +0.065]; V6 - SelfCheckGPT: +0.097 [+0.022, +0.175]*; V6 - semantic_entropy: +0.175 [+0.096, +0.255]*; V6 - highconf_contradicted: +0.042 [+0.005, +0.079]*; V6 - V3: -0.001 [-0.006, +0.004]; ctx_lr+fb - ctx_lr: +0.047 [+0.006, +0.091]*

Data efficiency (Amendment 1, descriptive): test AUROC vs labelled training sessions

| corpus | detector | 25 | 50 | 100 | 200 | all train |
|---|---|---|---|---|---|---|
| gpt-4o-mini (travel) | V6 | 0.798 | 0.837 | 0.848 | 0.873 | 0.876 |
| gpt-4o-mini (travel) | ctx_lr | 0.718 | 0.692 | 0.746 | 0.753 | 0.766 |
| gpt-4o-mini (travel) | ctx_lr+fb | 0.733 | 0.755 | 0.795 | 0.811 | 0.834 |
| 8B (travel) | V6 | 0.718 | 0.736 | 0.756 | 0.782 | 0.793 |
| 8B (travel) | ctx_lr | 0.616 | 0.668 | 0.690 | 0.708 | 0.733 |
| 8B (travel) | ctx_lr+fb | 0.661 | 0.692 | 0.739 | 0.751 | 0.776 |

Pooled over 2 corpora (stratified cluster bootstrap):

| id | quantity | estimate [95% CI] | licensed claim |
|---|---|---|---|
| H1 | AUROC(SelfCheckGPT) | 0.676 [0.632, 0.723] | report the AUROC; do not say 'near chance' |
| H1 | AUROC(semantic_entropy) | 0.549 [0.513, 0.594] | sampling detectors are near chance on user-specific hallucination |
| H2 | V6 - V6-prior | +0.078 [+0.048, +0.108]* | the user's implicit reactions add detectable signal beyond population base rates |
| H3 | V6 - ctx_lr | +0.085 [+0.043, +0.135]* | PGE beats the strongest context-only detector |
| H4 | V6 - SelfCheckGPT | +0.159 [+0.101, +0.207]* | PGE beats sampling-consistency detectors |
| H4 | V6 - semantic_entropy | +0.285 [+0.227, +0.331]* | PGE beats sampling-consistency detectors |
| H2b | ctx_lr+fb - ctx_lr | +0.055 [+0.023, +0.089]* | the user's reactions add detectable signal beyond the commitments themselves (supervised detector) |
| H4b | ctx_lr+fb - SelfCheckGPT | +0.129 [+0.066, +0.177]* | detectors on PGE observables beat sampling-consistency detectors |
| H4b | ctx_lr+fb - semantic_entropy | +0.255 [+0.189, +0.302]* | detectors on PGE observables beat sampling-consistency detectors |
| sec. | V6 - ctx_lr+fb | +0.030 [-0.000, +0.068] | reported as is |
| sec. | V6 - V3 | -0.003 [-0.010, -0.000]* | reported as is |

## Per-commitment check of Corollary 1 (user-dependent commitments; descriptive)

| corpus | n | wrong | all resamples agree | e1 = P(disagree|wrong) | e0 = P(disagree|right) | Cor. 1 bound | AUROC disagreement | AUROC one feedback bit |
|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini (travel) | 1659 | 0.306 | 0.928 | 0.171 | 0.028 | 0.597 | 0.572 | 0.745 |
| 8B (travel) | 1659 | 0.339 | 0.675 | 0.512 | 0.230 | 0.812 | 0.651 | 0.742 |

## Amendment 2: rules-aware analyses on L_U (oracle information: the simulator's rules and hidden-fact prior; not deployable)

| corpus | eta0 (Bayes, no feedback) | etaX (Bayes, shown reactions) | etaF (Bayes, feedback) | V6-session | LR-session | LR-session+fb |
|---|---|---|---|---|---|---|
| gpt-4o-mini (travel) | 0.786 | 0.859 | 0.862 | 0.797 | 0.661 | 0.772 |
| 8B (travel) | 0.944 | 0.964 | 0.963 | 0.925 | 0.912 | 0.929 |

Differences per corpus (session-clustered):

- **gpt-4o-mini (travel)**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.077 [+0.035, +0.120]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): +0.003 [-0.002, +0.007]; V6-session - LR-session+fb: +0.028 [-0.014, +0.075]; V6-session - eta0 (Bayes, no feedback): +0.017 [-0.044, +0.080]; LR-session+fb - LR-session: +0.115 [+0.065, +0.170]*
- **8B (travel)**: etaF (Bayes, feedback) - eta0 (Bayes, no feedback): +0.021 [+0.002, +0.041]*; etaF (Bayes, feedback) - etaX (Bayes, shown reactions): -0.001 [-0.002, +0.000]; V6-session - LR-session+fb: -0.003 [-0.027, +0.027]; V6-session - eta0 (Bayes, no feedback): -0.016 [-0.043, +0.009]; LR-session+fb - LR-session: +0.018 [-0.004, +0.041]

Pooled over 2 corpora:

| id | quantity | estimate [95% CI] | licensed claim |
|---|---|---|---|
| H5 | etaF (Bayes, feedback) - eta0 (Bayes, no feedback) | +0.048 [+0.025, +0.073]* | the user's reactions carry information beyond the population risk of the commitments |
| H6 | V6-session - LR-session+fb | +0.010 [-0.015, +0.040] | given the same information, the PGE posterior does not beat the supervised detector |
| sec. | V6-session - eta0 (Bayes, no feedback) | -0.004 [-0.033, +0.033] | reported as is |
| sec. | etaF (Bayes, feedback) - etaX (Bayes, shown reactions) | +0.001 [-0.001, +0.003] | reported as is |

runtime 2339s
