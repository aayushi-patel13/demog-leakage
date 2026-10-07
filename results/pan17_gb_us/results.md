Results (data/processed/pan17_gb_us, 20 encoder epochs, 100 attacker epochs; chance = 50.0)

| Measurement | Paper | Ours, paper protocol | Ours, held-out selection |
|---|---|---|---|
| Sentiment, encoder trained alone (acc) | - | 81.5 | 79.8 |
| Variety (GB vs US), encoder trained alone (acc) | - | 61.3 | 60.7 |
| Leakage: attacker on sentiment encoder | - | 56.1 | 54.7 |
| Adversarial (lambda=1): sentiment acc | - | 80.8 | 79.5 |
| Adversarial (lambda=1): online adversary acc | - | 48.2 | 50.8 |
| Adversarial (lambda=1): post-hoc attacker (leakage) | - | 55.7 | 53.0 |
| Adversarial (lambda=1): delta = attacker - adversary | - | 7.5 | 2.1 |

Paper protocol: best epoch chosen on the test set, as in the original code.
Held-out selection: epoch chosen on a separate validation set, then scored on test.
