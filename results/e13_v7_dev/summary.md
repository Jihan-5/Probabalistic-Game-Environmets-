# e13: V7 development (DEVELOPMENT corpora only, seed 42, protocol v2)

## 70B (dev), L_U

| detector | AUROC |
|---|---|
| V6 | 0.678 |
| V7-cells | 0.671 |
| V7 | 0.741 |
| V7m | 0.843 |
| ctx_lr+fb | 0.813 |
| intake_lr+fb | 0.844 |
| V6-session (oracle) | 0.904 |

- V7 - V6: +0.061 [+0.015, +0.109]
- V7 - ctx_lr+fb: -0.071 [-0.121, -0.022]
- V7 - intake_lr+fb: -0.104 [-0.155, -0.052]
- V7-cells - V6: -0.012 [-0.031, +0.008]
- V7m - V6: +0.164 [+0.113, +0.220]
- V7m - ctx_lr+fb: +0.032 [-0.016, +0.079]
- V7m - intake_lr+fb: -0.001 [-0.047, +0.047]

## 70B (dev), L

| detector | AUROC |
|---|---|
| V6 | 0.774 |
| V7-cells | 0.890 |
| V7 | 0.890 |
| V7m | 0.867 |
| ctx_lr+fb | 0.791 |
| intake_lr+fb | 0.923 |

- V7 - V6: +0.118 [+0.085, +0.152]
- V7 - ctx_lr+fb: +0.101 [+0.057, +0.147]
- V7 - intake_lr+fb: -0.033 [-0.068, +0.001]
- V7-cells - V6: +0.118 [+0.085, +0.152]
- V7m - V6: +0.093 [+0.049, +0.137]
- V7m - ctx_lr+fb: +0.076 [+0.028, +0.125]
- V7m - intake_lr+fb: -0.058 [-0.095, -0.023]

## 8B (dev), L_U

| detector | AUROC |
|---|---|
| V6 | 0.722 |
| V7-cells | 0.722 |
| V7 | 0.744 |
| V7m | 0.789 |
| ctx_lr+fb | 0.630 |
| intake_lr+fb | 0.785 |
| V6-session (oracle) | 0.956 |

- V7 - V6: +0.037 [-0.044, +0.114]
- V7 - ctx_lr+fb: +0.115 [+0.029, +0.201]
- V7 - intake_lr+fb: -0.050 [-0.139, +0.035]
- V7-cells - V6: +0.019 [-0.022, +0.059]
- V7m - V6: +0.077 [+0.001, +0.150]
- V7m - ctx_lr+fb: +0.155 [+0.039, +0.273]
- V7m - intake_lr+fb: -0.011 [-0.122, +0.096]

## 8B (dev), L

| detector | AUROC |
|---|---|
| V6 | 0.734 |
| V7-cells | 0.869 |
| V7 | 0.869 |
| V7m | 0.861 |
| ctx_lr+fb | 0.660 |
| intake_lr+fb | 0.817 |

- V7 - V6: +0.136 [+0.073, +0.203]
- V7 - ctx_lr+fb: +0.206 [+0.112, +0.312]
- V7 - intake_lr+fb: +0.056 [-0.027, +0.144]
- V7-cells - V6: +0.136 [+0.073, +0.203]
- V7m - V6: +0.135 [+0.074, +0.202]
- V7m - ctx_lr+fb: +0.205 [+0.110, +0.310]
- V7m - intake_lr+fb: +0.055 [-0.026, +0.140]

runtime 170s
