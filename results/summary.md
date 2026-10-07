# Results summary

Held-out selection throughout; ± is a 95% interval for each accuracy (chance = 50.0).

| | Paper | Tweets, race (clean data) | Tweets, race (original pipeline as written) | Reddit, NG vs US | Reddit, NG vs US, topic words masked |
|---|---|---|---|---|---|
| Main task, encoder trained alone | 67.4 | 70.2 ± 0.9 | 71.6 ± 0.9 | 87.3 ± 1.4 | 86.1 ± 1.4 |
| Protected attribute, encoder trained for it | 83.9 | 84.5 ± 0.7 | 88.8 ± 0.6 | 72.2 ± 1.9 | 70.6 ± 1.9 |
| Leakage: attacker on the main-task encoder | 64.5 | 64.1 ± 0.9 | 70.8 ± 0.9 | 54.5 ± 2.1 | 56.6 ± 2.1 |
| Adversarial training: main task | 64.7 | 70.9 ± 0.9 | 71.4 ± 0.9 | 85.4 ± 1.5 | 86.7 ± 1.4 |
| Adversarial training: online adversary | - | 51.2 ± 1.0 | 52.9 ± 1.0 | 49.7 ± 2.1 | 50.3 ± 2.1 |
| Adversarial training: post-hoc attacker (leakage) | 56.0 | 65.0 ± 0.9 | 70.2 ± 0.9 | 56.0 ± 2.1 | 54.1 ± 2.1 |
| Adversarial training: epoch chosen on validation | - | 5 of 20 | 6 of 20 | 5 of 20 | 15 of 20 |

Test sizes: Tweets, race (clean data): 10,000 test examples; Tweets, race (original pipeline as written): 10,000 test examples; Reddit, NG vs US: 2,220 test examples; Reddit, NG vs US, topic words masked: 2,220 test examples.

## Adversarial training, repeated (tweets, clean data, lambda = 1)

| Run | Epochs | Main task | Online adversary | Post-hoc attacker (leakage) |
|---|---|---|---|---|
| seed 16 (main run) | 20 (best on val: 5) | 70.9 ± 0.9 | 51.2 ± 1.0 | 65.0 ± 0.9 |
| seed 17 | 20 (best on val: 4) | 70.8 ± 0.9 | 50.4 ± 1.0 | 62.9 ± 0.9 |
| seed 18 | 20 (best on val: 5) | 70.3 ± 0.9 | 54.5 ± 1.0 | 64.1 ± 0.9 |
| long run, seed 16 | after 20 | 69.0 ± 0.9 | 51.9 ± 1.0 | 63.8 ± 0.9 |
| long run, seed 16 | after 40 | 68.0 ± 0.9 | 52.2 ± 1.0 | 62.7 ± 0.9 |
| long run, seed 16 | after 60 | 67.5 ± 0.9 | 51.6 ± 1.0 | 62.0 ± 1.0 |

Across 3 seeds: leakage 64.0 (sd 1.0), online adversary 52.0 (sd 2.2), main task 70.7 (sd 0.3).

