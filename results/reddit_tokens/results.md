Results (data/reddit/processed/tokens, 20 encoder epochs, 100 attacker epochs; chance = 50.0)

| Measurement | Paper | Ours, paper protocol | Ours, held-out selection |
|---|---|---|---|
| Sentiment, encoder trained alone (acc) | - | 87.3 | 87.3 |
| Variety (NG vs US), encoder trained alone (acc) | - | 73.6 | 72.2 |
| Leakage: attacker on sentiment encoder | - | 56.2 | 54.5 |
| Adversarial (lambda=1): sentiment acc | - | 86.9 | 85.4 |
| Adversarial (lambda=1): online adversary acc | - | 50.8 | 49.7 |
| Adversarial (lambda=1): post-hoc attacker (leakage) | - | 57.4 | 56.0 |
| Adversarial (lambda=1): delta = attacker - adversary | - | 6.6 | 6.3 |

Paper protocol: best epoch chosen on the test set, as in the original code.
Held-out selection: epoch chosen on a separate validation set, then scored on test.
