Results (data/processed/sent_race_faithful, 20 encoder epochs, 100 attacker epochs; chance = 50.0)

| Measurement | Paper | Ours, paper protocol | Ours, held-out selection |
|---|---|---|---|
| Sentiment, encoder trained alone (acc) | 67.4 | 72.1 | 71.6 |
| Race, encoder trained alone (acc) | 83.9 | 89.2 | 88.8 |
| Leakage: attacker on sentiment encoder | 64.5 | 71.2 | 70.8 |
| Adversarial (lambda=1): sentiment acc | 64.7 | 71.4 | 71.4 |
| Adversarial (lambda=1): online adversary acc | - | 52.9 | 52.9 |
| Adversarial (lambda=1): post-hoc attacker (leakage) | 56.0 | 70.4 | 70.2 |
| Adversarial (lambda=1): delta = attacker - adversary | 5.0 | 17.6 | 17.4 |

Paper protocol: best epoch chosen on the test set, as in the original code.
Held-out selection: epoch chosen on a separate validation set, then scored on test.
