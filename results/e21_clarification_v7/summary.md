# e21: clarification with the V7 picker (Amendment 7) vs the registered run

paired sessions: 120

| condition | questions/sess | L_U rate | wrong conf. user-dep./sess (primary) | L rate |
|---|---|---|---|---|
| V7 posterior@1 | 1.00 | 0.242 | 0.367 | 0.117 |
| V7 posterior@2 | 2.00 | 0.167 | 0.333 | 0.058 |
| cached | 0.00 | 0.467 | 0.642 | 0.350 |
| posterior@1 | 1.00 | 0.383 | 0.575 | 0.142 |
| posterior@2 | 2.00 | 0.358 | 0.542 | 0.025 |
| random@1 | 1.00 | 0.392 | 0.583 | 0.167 |
| lowconf@1 | 1.00 | 0.458 | 0.650 | 0.150 |
| no_question@1 | 0.00 | 0.492 | 0.700 | 0.300 |
| no_question@2 | 0.00 | 0.475 | 0.667 | 0.275 |
| detailed_questions | 4.29 | 0.042 | 0.108 | 0.042 |
| rerun_full | 0.00 | 0.483 | 0.617 | 0.275 |

Paired bootstrap over sessions (2000), A - B, negative = A better:

| id | A | B | d wrong conf. user-dep./sess | d L_U rate | d L rate |
|---|---|---|---|---|---|
| H9 | V7 posterior@1 | random@1 | -0.217 [-0.325, -0.108] | -0.150 [-0.233, -0.067] | -0.050 [-0.108, +0.008] |
| -- | V7 posterior@1 | no_question@1 | -0.333 [-0.458, -0.208] | -0.250 [-0.342, -0.158] | -0.183 [-0.267, -0.100] |
| -- | V7 posterior@1 | lowconf@1 | -0.283 [-0.400, -0.158] | -0.217 [-0.308, -0.125] | -0.033 [-0.100, +0.025] |
| -- | V7 posterior@1 | posterior@1 | -0.208 [-0.308, -0.108] | -0.142 [-0.225, -0.067] | -0.025 [-0.075, +0.025] |
| -- | V7 posterior@2 | no_question@2 | -0.333 [-0.484, -0.175] | -0.308 [-0.417, -0.192] | -0.217 [-0.308, -0.133] |
| -- | V7 posterior@2 | detailed_questions | +0.225 [+0.125, +0.333] | +0.125 [+0.058, +0.200] | +0.017 [-0.033, +0.067] |
| -- | V7 posterior@2 | posterior@2 | -0.208 [-0.317, -0.108] | -0.192 [-0.275, -0.117] | +0.033 [+0.000, +0.075] |
| -- | V7 posterior@1 | cached | -0.275 [-0.400, -0.150] | -0.225 [-0.317, -0.125] | -0.233 [-0.317, -0.150] |
