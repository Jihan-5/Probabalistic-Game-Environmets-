# e22: frozen V7 on the LLM-patient v2 corpus (exploratory; V7 is not registered here)

## patient_v2: L_U

| corpus | V6 | V7 | ctx_lr | ctx_lr+fb | intake_lr+fb |
|---|---|---|---|---|---|
| 70B LLM-patient v2 (profile) | 0.644 | 0.719 | 0.729 | 0.733 | 0.775 |

- 70B LLM-patient v2 (profile): V7 - V6 +0.068 [+0.025, +0.109]; V7 - intake_lr+fb -0.060 [-0.115, -0.003]; V7 - ctx_lr+fb -0.018 [-0.076, +0.037]

## patient_v2: L

| corpus | V6 | V7 | ctx_lr | ctx_lr+fb | intake_lr+fb |
|---|---|---|---|---|---|
| 70B LLM-patient v2 (profile) | 0.743 | 0.732 | 0.769 | 0.731 | 0.865 |

- 70B LLM-patient v2 (profile): V7 - V6 -0.012 [-0.023, -0.000]; V7 - intake_lr+fb -0.137 [-0.207, -0.071]; V7 - ctx_lr+fb -0.001 [-0.058, +0.057]

## patient_v2: L_U label budget (descriptive)

- 70B LLM-patient v2 (profile): V7@100 0.661, V7@200 0.677, V7@25 0.654, V7@50 0.638, ctx_lr+fb@100 0.667, ctx_lr+fb@200 0.707, ctx_lr+fb@25 0.639, ctx_lr+fb@50 0.670, intake_lr+fb@100 0.706, intake_lr+fb@200 0.747, intake_lr+fb@25 0.666, intake_lr+fb@50 0.706

## patient_v2: per-commitment q_t of V7

- 70B LLM-patient v2 (profile): AUROC 0.691, ECE 0.289 (n per split 232)

runtime 185s
