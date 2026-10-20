# e7: stress tests of the label-event posterior (V3), offline, $0

Date 2026-10-19. No API calls. Every number below comes from running the e6 protocol unchanged on cached trajectories: `split_task_ids(ids, 0.7, seed)` with seeds 42/123/456, fit on train, test scored once, mean AUROC over seeds, and a paired bootstrap (2000 resamples, rng 0) of AUROC differences with the per-seed test sets pooled. Each CI is reported two ways. **item** is the e6 protocol and resamples rows. **task** resamples unique task_ids, so every pooled row of a task enters together. `*` means the CI excludes 0. "Best baseline" is the baseline with the highest mean AUROC, picked post hoc per row, which favours the baselines.

**Truth use (analysis only).** Per-turn correctness is reconstructed from the generator seed stream (`eval.e6_ceiling.load_with_truth`). It is used only to (a) define alternative labels (E7a) and (b) regenerate an alternative detector-visible feedback channel (E7b, E7c, redraw supplement). Detectors always get copies whose `eval_label` is masked to 0. Train labels reach them only through `fit()`.

**Reproduction checks (all pass).**
* E7a at (0.6, 0.5) reproduces the e6 headlines exactly: 70B +0.066 [+0.021, +0.110]; 8B +0.059 [-0.037, +0.159].
* E7d full V3 gives 0.775 / 0.728.
* The E7c oracle on the recorded channel gives 0.824 / 0.817, matching e6_ceiling.
* E7e item-level CIs equal e6 to the third decimal.

Nothing in posterior.py or the e5/e6 files was edited. Hyperparameters were fixed before any run: k=5, bucket edges 0.6/0.65/0.75, eps grid 0.05..0.45.

## E7a: label-definition sensitivity (`results/e7_label_sensitivity/summary.txt`)

Labels are recomputed with the eval_label_v2 rule for each (high, frac). Detectors use matching settings: V3/V1 with the same high and frac, and HighConfContradicted with threshold=high. dag_logreg is retrained per label, and its fixed 0.6 high-confidence feature is left untouched. Two definitions were skipped for having too few positives: 70B (0.5, 0.7) at 2.0% and 70B (0.6, 0.7) at 3.5%.

| corpus | high | frac | hall rate | V3 | best baseline | V3 - best (item) | V3 - best (task) |
|---|---|---|---|---|---|---|---|
| 70B | 0.5 | 0.3 | 0.618 | 0.761 | NSF 0.710 | +0.046 [+0.004, +0.088]* | +0.046 [-0.004, +0.095] |
| 70B | 0.5 | 0.5 | 0.230 | 0.811 | HCC 0.741 | +0.066 [+0.030, +0.102]* | +0.067 [+0.022, +0.110]* |
| 70B | 0.6 | 0.3 | 0.517 | 0.781 | HCC 0.709 | +0.073 [+0.030, +0.118]* | +0.073 [+0.017, +0.130]* |
| 70B | 0.6 | 0.5 | 0.250 | 0.775 | HCC 0.711 | +0.066 [+0.021, +0.110]* | +0.066 [+0.009, +0.127]* |
| 70B | 0.7 | 0.3 | 0.290 | 0.827 | HCC 0.709 | +0.119 [+0.073, +0.166]* | +0.120 [+0.064, +0.176]* |
| 70B | 0.7 | 0.5 | 0.225 | 0.845 | HCC 0.715 | +0.129 [+0.086, +0.175]* | +0.129 [+0.074, +0.184]* |
| 70B | 0.7 | 0.7 | 0.083 | 0.948 | dag_logreg 0.829 | +0.120 [+0.074, +0.174]* | +0.120 [+0.059, +0.185]* |
| 8B | 0.5 | 0.3 | 0.610 | 0.791 | HCC 0.732 | +0.044 [-0.022, +0.109] | +0.040 [-0.042, +0.117] |
| 8B | 0.5 | 0.5 | 0.390 | 0.754 | SelfCheck 0.649 | +0.098 [-0.013, +0.208] | +0.100 [-0.041, +0.244] |
| 8B | 0.5 | 0.7 | 0.135 | 0.852 | dag_logreg 0.803 | +0.047 [-0.026, +0.122] | +0.049 [-0.039, +0.139] |
| 8B | 0.6 | 0.3 | 0.625 | 0.779 | SelfCheck 0.692 | +0.066 [-0.052, +0.187] | +0.066 [-0.077, +0.215] |
| 8B | 0.6 | 0.5 | 0.400 | 0.728 | SelfCheck 0.669 | +0.059 [-0.037, +0.158] | +0.059 [-0.071, +0.188] |
| 8B | 0.6 | 0.7 | 0.110 | 0.855 | HCC 0.844 | +0.002 [-0.093, +0.110] | +0.000 [-0.125, +0.151] |
| 8B | 0.7 | 0.3 | 0.465 | 0.764 | HCC 0.694 | +0.069 [+0.004, +0.135]* | +0.067 [-0.017, +0.150] |
| 8B | 0.7 | 0.5 | 0.395 | 0.817 | SelfCheck 0.714 | +0.102 [-0.004, +0.208] | +0.100 [-0.037, +0.240] |
| 8B | 0.7 | 0.7 | 0.140 | 0.843 | dag_logreg 0.812 | +0.043 [-0.072, +0.159] | +0.047 [-0.098, +0.171] |

Takeaways:
* **70B: the advantage holds on all 7 evaluable definitions** (item CI excludes 0 on 7/7, task CI on 6/7). It grows to about +0.12 at high=0.7.
* **8B: V3 has the best point estimate on 9/9, but the item CI excludes 0 on only 1/9.** At (0.6, 0.7) the advantage disappears (+0.002), and V1 beats V3 there (0.872 vs 0.855).
* Caveat in V3's favour at high≠0.6: dag_logreg keeps its built-in 0.6 feature, which is not adapted to the new threshold.

## E7b: feedback-channel misspecification (`results/e7_channel_misspec/summary.txt`)

This is a detector-side stress test: the feedback is regenerated from truth, labels stay the same, and train and test use the same channel. Each setting uses 3 channel draws × 3 split seeds. **Caveat: the agent's in-context feedback was the original eps=0.25 channel. The agent never reacted to these channels.** V3 still assumes a symmetric flip channel. SelfCheckGPT reads resamples only, so it is channel-independent (0.504 on 70B, 0.669 on 8B).

| channel | 70B V3 | 70B best baseline | 70B V3 - best (item) | 8B V3 | 8B best baseline | 8B V3 - best (item) |
|---|---|---|---|---|---|---|
| recorded | 0.775 | HCC 0.711 | +0.066 [+0.021, +0.110]* | 0.728 | SelfCheck 0.669 | +0.059 [-0.037, +0.159] |
| sym eps 0.10 | 0.890 | HCC 0.879 | +0.011 [-0.002, +0.024] | 0.887 | HCC 0.862 | +0.027 [+0.010, +0.045]* |
| sym eps 0.25 redraw | 0.793 | dag 0.746 | +0.051 [+0.028, +0.075]* | 0.779 | HCC 0.742 | +0.034 [+0.002, +0.065]* |
| sym eps 0.40 | 0.690 | dag 0.672 | +0.014 [-0.023, +0.050] | 0.648 | SelfCheck 0.669 | **-0.023** [-0.081, +0.033] |
| asym 0.9 / 0.35 | 0.792 | dag 0.774 | +0.017 [-0.009, +0.042] | 0.782 | HCC 0.772 | +0.008 [-0.017, +0.032] |
| asym 0.6 / 0.10 | 0.824 | dag 0.786 | +0.037 [+0.020, +0.053]* | 0.816 | HCC 0.753 | +0.055 [+0.029, +0.082]* |
| attr eps ~ U(0.1, 0.4) | 0.779 | dag 0.754 | +0.025 [-0.002, +0.050] | 0.751 | HCC 0.698 | +0.052 [+0.021, +0.084]* |
| recorded + drop 0.3 | 0.760 | dag 0.678 | +0.083 [+0.055, +0.110]* | 0.703 | SelfCheck 0.669 | +0.031 [-0.029, +0.090] |
| recorded + drop 0.5 | 0.747 | dag 0.685 | +0.067 [+0.037, +0.097]* | 0.669 | SelfCheck 0.669 | **-0.004** [-0.064, +0.056] |
| lazy user (0.6, no +1) | **0.425** | dag 0.829 | **-0.403 [-0.444, -0.360]*** | **0.576** | dag 0.797 | **-0.213 [-0.279, -0.147]*** |

Takeaways:
* **V3 has the best point estimate under every two-sided channel except 8B sym eps=0.40** (SelfCheckGPT 0.669 vs 0.648, n.s.). That covers the asymmetric, attribute-dependent and extra-missingness channels. It wins significantly on 5/9 non-lazy 70B channels and 4/9 on 8B. The task-level CIs (in the per-experiment file) are wider. On 8B, under heavy noise (eps 0.40) or heavy missingness (drop 0.5), it falls to the channel-free SelfCheckGPT level.
* **V3 fails under the one-sided "lazy user" channel**: 0.425 on 70B, below chance. Mechanism (`lazy_diagnosis.txt`): every observed signal is -1, so the method-of-moments prior clips to 0.99, eps runs to the grid edge 0.45 and the tail probability saturates at about 1. Silence carries information here (missing not at random), and V3 treats it as ignorable. A trivial statistic, the fraction of high-confidence turns contradicted, reaches 0.80 / 0.82.
* **Value conditioning on 8B is not robust to redrawing the channel**: V3 - V1 is +0.043* on the recorded draw but -0.006 on the eps=0.25 redraw. On 70B it stays positive (+0.041 recorded, +0.030 redraw).

**Redraw supplement** (`results/e7_channel_misspec/redraw_variance.txt`) compares the recorded channel with 20 redraws of the same eps=0.25 channel.

| | V3 recorded / redraw mean (sd) | margin vs best baseline, recorded / redraw mean (sd) | margin > 0 |
|---|---|---|---|
| 70B | 0.775 / 0.804 (0.017), recorded at 5th pct | +0.065 / +0.054 (0.018), recorded at 85th pct | 20/20 |
| 8B | 0.728 / 0.759 (0.043) | +0.059 / **+0.020** (0.031), recorded at 95th pct | 16/20 |

* **70B:** the margin is stable across redraws, with a minimum of +0.021.
* **8B:** the headline margin is optimistic. On the recorded draw, HCC and dag_logreg both fall at the 0th percentile of their redraw distributions, and the expected margin is about +0.02.
* The recorded channel is coupled to the agent on 70B. After a -1, the agent's next commitment is high-confidence 42–44% of the time, against 61–68% after a +1, regardless of truth. Recorded and redrawn channels are therefore not exchangeable on 70B. 8B shows little coupling.

## E7c: oracle ceiling vs noise (`results/e7_ceiling_noise/summary.txt`, `table_{70B,8B}.csv`, `ceiling_vs_noise.png`)

The feedback is a regenerated symmetric channel, averaged over 3 draws. The oracle is e6_ceiling's: in-sample (attribute, value, bucket) priors from truth plus the true eps, reported on the test splits. Learned detectors follow the e6 protocol.

| eps | 70B oracle | 70B V3 | 70B V3 - best (item) | 8B oracle | 8B V3 | 8B V3 - best (item) |
|---|---|---|---|---|---|---|
| 0.00 | 0.986 | 0.960 | +0.002 (HCC) | 0.990 | 0.981 | +0.023* (HCC) |
| 0.05 | 0.948 | 0.932 | +0.012* | 0.940 | 0.935 | +0.025* |
| 0.10 | 0.916 | 0.901 | +0.019* | 0.916 | 0.897 | +0.036* |
| 0.15 | 0.879 | 0.851 | +0.030* | 0.898 | 0.872 | +0.035* |
| 0.20 | 0.856 | 0.827 | +0.060* (dag) | 0.858 | 0.822 | +0.038* |
| 0.25 | 0.847 | 0.817 | +0.045* | 0.826 | 0.777 | +0.031 |
| 0.30 | 0.812 | 0.781 | +0.067* | 0.808 | 0.716 | +0.003 |
| 0.35 | 0.794 | 0.735 | +0.051* | 0.787 | 0.666 | -0.010 |
| 0.40 | 0.786 | 0.704 | +0.022 | 0.753 | 0.596 | **-0.074* (SelfCheck)** |
| 0.45 | 0.781 | 0.599 | **-0.070* (dag)** | 0.736 | 0.582 | **-0.086* (SelfCheck)** |

Takeaways:
* V3 stays within 0.015–0.031 of the oracle up to eps=0.30 on 70B, and within 0.05 up to eps=0.25 on 8B.
* **V3's lead peaks at moderate noise (eps 0.15–0.35) and disappears at both ends.** At eps 0 it ties HCC. At eps ≥ 0.40 the learned method of moments degrades: the gap to the oracle grows to 0.15–0.18, and V3 loses to dag_logreg on 70B and to SelfCheckGPT on 8B. The 8B losses are significant at item level only.
* At eps=0, V3's eps grid floor of 0.05 (fixed a priori) is a minor handicap.

## E7d: ablation (`results/e7_ablation/summary.txt`)

All differences are AUROC(variant) - AUROC(full V3), with item CIs. Task CIs are in the file and agree in sign and significance except where noted.

| variant | 70B AUROC | 70B diff | 8B AUROC | 8B diff |
|---|---|---|---|---|
| full V3 | 0.775 | | 0.728 | |
| - value conditioning (V1) | 0.734 | -0.041* | 0.678 | -0.043* |
| flat prior (global rate) | 0.716 | -0.057* | 0.625 | -0.093* |
| - event targeting [a] mean posterior, own eps | 0.744 | -0.040 [-0.083, +0.001] | 0.677 | -0.053* (task n.s.) |
| - event targeting [b] mean posterior, V3 fit | 0.772 | -0.003 [-0.027, +0.019] | 0.695 | -0.039 [-0.078, +0.001] |
| - high-conf restriction [a] own eps | 0.714 | -0.064* | 0.584 | -0.157* |
| - high-conf restriction [b] V3 fit | 0.718 | -0.061* | 0.602 | -0.131* |
| eps fixed 0.25 (no labels used) | 0.775 | +0.000 | 0.742 | **+0.014*** |
| eps fixed 0.10 (misspecified) | 0.745 | -0.030 [-0.061, +0.001] | 0.668 | -0.056* |
| eps fixed 0.40 (misspecified) | 0.745 | -0.037* | 0.720 | -0.020 |
| + samples (V4) | 0.773 | -0.003 | 0.736 | +0.007 |
| shrinkage k=1 | 0.779 | +0.003 | 0.741 | +0.013 |
| shrinkage k=20 | 0.765 | -0.009* | 0.707 | -0.015 |

Takeaways:
* **The gain comes from the prior structure (value, attribute, bucket) and from restricting the score to high-confidence turns.** On 70B the Poisson-binomial tail itself adds nothing over a mean posterior once the fit is fixed (-0.003, n.s.). On 8B the tail is worth about +0.04, which is borderline.
* **The label-based eps selection is unnecessary.** Fixing eps at the known channel value 0.25 matches V3 on 70B, where it is identical because V3 picked 0.25 on all 3 seeds, and beats it on 8B (+0.014*), which makes V3 a fully unsupervised detector given eps. Misspecifying eps costs 0.02–0.06.
* V4 and k=1 differ from V3 only within noise. k=5 is not the best k, so it was not cherry-picked.

## E7e: dbbench 70B pooled, task-cluster bootstrap (`results/e7_dbbench_cluster/summary.txt`)

The pooled test set has 315 rows from 81 unique tasks, with 3 to 9 rows per task.

| pge_evidence_nb - | item-level CI (e6) | task-level CI | width ratio |
|---|---|---|---|
| selfcheckgpt | +0.015 [-0.022, +0.051] | +0.016 [-0.049, +0.079] | 1.75 |
| dag_logreg | +0.050 [+0.000, +0.101]* | +0.050 [-0.015, +0.113] | 1.27 |
| semantic_entropy | +0.054 [+0.009, +0.097]* | +0.055 [-0.023, +0.133] | 1.75 |
| highconf_contradicted | +0.198 [+0.141, +0.253]* | +0.200 [+0.096, +0.294]* | 1.75 |
| neg_signal_fraction | +0.091 [+0.039, +0.141]* | +0.093 [+0.008, +0.177]* | 1.67 |

Takeaway: **once tasks are clustered, the e6 dbbench wins over dag_logreg and semantic_entropy are no longer significant.** EvidenceNB ties SelfCheckGPT either way and beats only HCC and NSF. The item-level CIs in e6 treat the three generation copies of a task as independent and should not be cited.

## All variants run (nothing omitted)
* E7a: 18 label definitions × 6 detectors, with 2 definitions skipped (<5% positives).
* E7b: 10 channels × 2 corpora × 3 draws × 3 seeds × 6 detectors, plus the 20-redraw supplement (4 detectors) and the lazy-user diagnosis.
* E7c: 10 eps levels × 3 draws × 2 corpora (oracle + 5 detectors).
* E7d: 13 variants × 2 corpora.
* E7e: 6 detectors.

Process notes:
* The first E7a run loaded rows in truth-loader order rather than cache order. Point estimates were identical, but the CI bound differed by 0.001 from e6 (+0.022 vs +0.021). The loader now uses cache order. Only the corrected run is reported.
* E7b and E7c were rerun after adding SelfCheckGPT as a channel-free reference and CIs to E7c. Every earlier column reproduced exactly (fixed seeds).

## Runtimes (2 cores, sequential)
* E7a 61 s
* E7b 165 s
* redraw supplement 82 s
* lazy diagnosis 2 s
* E7c 167 s
* E7d 80 s
* E7e 6 s

Total about 9.5 min.

## Reproduce (repo root, Python 3.13, numpy/scipy; matplotlib optional for the PNG)
```
python3 -m eval.e7_label_sensitivity      # E7a -> results/e7_label_sensitivity/summary.txt
python3 -m eval.e7_channel_misspec        # E7b -> results/e7_channel_misspec/summary.txt
python3 -m eval.e7_redraw_variance       # E7b supplement -> results/e7_channel_misspec/redraw_variance.txt
python3 -m eval.e7_lazy_diagnosis        # E7b diagnosis -> results/e7_channel_misspec/lazy_diagnosis.txt
python3 -m eval.e7_ceiling_noise          # E7c -> results/e7_ceiling_noise/{summary.txt,table_*.csv,ceiling_vs_noise.png}
python3 -m eval.e7_ablation               # E7d -> results/e7_ablation/summary.txt
python3 -m eval.e7_dbbench_cluster        # E7e -> results/e7_dbbench_cluster/summary.txt
```
Shared harness: `eval/e7_common.py`.
