Results (data/processed/sent_race, 20 encoder epochs, 100 attacker epochs; chance = 50.0)

| Measurement | Paper | Ours, paper protocol | Ours, held-out selection |
|---|---|---|---|
| Sentiment, encoder trained alone (acc) | 67.4 | 70.7 | 70.2 |
| Race, encoder trained alone (acc) | 83.9 | 85.4 | 84.5 |
| Leakage: attacker on sentiment encoder | 64.5 | 65.4 | 64.1 |
| Adversarial (lambda=1): sentiment acc | 64.7 | 70.9 | 70.9 |
| Adversarial (lambda=1): online adversary acc | - | 51.2 | 51.2 |
| Adversarial (lambda=1): post-hoc attacker (leakage) | 56.0 | 65.0 | 65.0 |
| Adversarial (lambda=1): delta = attacker - adversary | 5.0 | 13.8 | 13.8 |

Paper protocol: best epoch chosen on the test set, as in the original code.
Held-out selection: epoch chosen on a separate validation set, then scored on test.
