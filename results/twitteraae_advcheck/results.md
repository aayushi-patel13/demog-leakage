Adversarial training check (data/processed/sent_race, lambda = 1; held-out selection; chance = 50.0)

| Run | Epochs | Sentiment acc | Online adversary acc | Post-hoc attacker (leakage) |
|---|---|---|---|---|
| seed 16 (main run) | 20 (best on val) | 70.9 | 51.2 | 65.0 |
| seed 17 | 20 (best on val: 4) | 70.8 | 50.4 | 62.9 |
| seed 18 | 20 (best on val: 5) | 70.3 | 54.5 | 64.1 |
| long run, seed 16 | after 20 | 69.0 | 51.9 | 63.8 |
| long run, seed 16 | after 40 | 68.0 | 52.2 | 62.7 |
| long run, seed 16 | after 60 | 67.5 | 51.6 | 62.0 |

Across 3 seeds: leakage 64.0 (sd 1.0), online adversary 52.0 (sd 2.2), sentiment 70.7 (sd 0.3).
