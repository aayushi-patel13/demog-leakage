# Authors' original code (Python 2.7, DyNet) on the faithful TwitterAAE data

Best epoch chosen on the test set, as the original code does; 10,000 test tweets; chance = 50.

| | Paper | Original code | PyTorch port, same data |
|---|---|---|---|
| Sentiment only (main task) | 67.4 | 73.1 | 72.1 |
| Leakage: attacker on the sentiment encoder | 64.5 | 78.4 | 71.2 |
| Adversarial training: main task | 64.7 | 69.8 | 71.4 |
| Adversarial training: online adversary | 51.0 | 50.1 | 52.9 |
| Adversarial training: attacker on best model | 56.0 | 67.9 | 70.4 |

Epochs completed: baseline 2, adversarial 1; attacker epochs: on the sentiment encoder 100, on the adversarial encoder 30. Epochs chosen: sentiment 2, adversarial 1. The paper's adversary figure is its attacker score minus its reported gap of 5.0.

Adversarial run, per epoch (test): epoch 1: task 69.8, adversary 50.1

## Epoch-matched comparison: same data, same number of epochs

| | Original code | PyTorch port |
|---|---|---|
| Sentiment only, best of first 2 epoch(s) | 73.1 | 69.4 |
| Leakage from that encoder | 78.4 | 71.9 |
| Adversarial, best of first 1 epoch(s): sentiment | 69.8 | 68.8 |
| Adversarial: online adversary | 50.1 | 55.5 |
| Adversarial: attacker | 67.9 | 69.2 |

Both select the best test epoch and report the attacker's best test epoch, as the original code does. The port trains on 156,000 tweets because it holds out a validation set; the original trains on 166,000. The 10,000 test tweets are the same.

## Same attacker budget: best of the first 30 attacker epochs

The attacker on the original adversarial model ran for 30 epochs, because the Codespace stopped longer runs, so every attacker here is scored over its first 30 epochs.

| | Original code | PyTorch port |
|---|---|---|
| Leakage from the sentiment-only encoder | 77.8 | 70.3 |
| Attacker on the adversarial encoder | 67.9 | 67.7 |
