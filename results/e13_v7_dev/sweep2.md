# e13 sweep 2 (DEVELOPMENT corpora only, L_U)

## 70B (dev)

- V7m k2: 0.883
- V7m k3: 0.883
- V7m k5: 0.882
- intake_lr+fb: 0.844
- V7m k2 - intake_lr+fb: +0.037 [-0.002, +0.078]
- V7m k3 - intake_lr+fb: +0.036 [-0.003, +0.078]
- V7m k5 - intake_lr+fb: +0.037 [-0.003, +0.078]

label budget:
- V7m k5 @ 25: 0.658
- intake_lr+fb @ 25: 0.651
- V7m k5 @ 50: 0.684
- intake_lr+fb @ 50: 0.721
- V7m k5 @ 100: 0.751
- intake_lr+fb @ 100: 0.773
- V7m k5 @ 200: 0.805
- intake_lr+fb @ 200: 0.804

## 8B (dev)

- V7m k2: 0.862
- V7m k3: 0.811
- V7m k5: 0.805
- intake_lr+fb: 0.785
- V7m k2 - intake_lr+fb: +0.060 [-0.025, +0.149]
- V7m k3 - intake_lr+fb: +0.024 [-0.066, +0.115]
- V7m k5 - intake_lr+fb: +0.009 [-0.096, +0.109]

label budget:
- V7m k5 @ 25: 0.679
- intake_lr+fb @ 25: 0.666
- V7m k5 @ 50: 0.716
- intake_lr+fb @ 50: 0.686
- V7m k5 @ 100: 0.764
- intake_lr+fb @ 100: 0.757

