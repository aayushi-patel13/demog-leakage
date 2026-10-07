Results (data/reddit/processed/masked, 20 encoder epochs, 100 attacker epochs; chance = 50.0)

| Measurement | Paper | Ours, paper protocol | Ours, held-out selection |
|---|---|---|---|
| Sentiment, encoder trained alone (acc) | - | 87.8 | 86.1 |
| Variety (NG vs US), encoder trained alone (acc) | - | 71.4 | 70.6 |
| Leakage: attacker on sentiment encoder | - | 57.5 | 56.6 |
| Adversarial (lambda=1): sentiment acc | - | 86.8 | 86.7 |
| Adversarial (lambda=1): online adversary acc | - | 52.1 | 50.3 |
| Adversarial (lambda=1): post-hoc attacker (leakage) | - | 57.0 | 54.1 |
| Adversarial (lambda=1): delta = attacker - adversary | - | 4.9 | 3.8 |

Paper protocol: best epoch chosen on the test set, as in the original code.
Held-out selection: epoch chosen on a separate validation set, then scored on test.
