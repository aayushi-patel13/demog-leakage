## TwitterAAE: the replication data

The paper uses 44,000 tweets per quadrant (41,500 train + 2,500 test).

### Clean preprocessing (main replication) (`data/processed/sent_race`)

| Quadrant | Available in corpus | Written | Distinct | Also under the opposite label | Train / val / test |
|---|---:|---:|---:|---:|---|
| pos_aa | 72,216 | 46,500 | 46,500 | 0 | 26,108 / 2,500 / 2,500 |
| pos_wh | 1,478,256 | 46,500 | 46,500 | 0 | 26,108 / 2,500 / 2,500 |
| neg_aa | 38,739 | 31,108 (short of 44,000) | 31,108 | 0 | 26,108 / 2,500 / 2,500 |
| neg_wh | 733,037 | 46,500 | 46,500 | 0 | 26,108 / 2,500 / 2,500 |

Test sentences also in training (paper split): 0. Mean tokens per tweet: pos_aa 10.49, pos_wh 11.18, neg_aa 10.3, neg_wh 11.58.

### Original preprocessing as written (ablation) (`data/processed/sent_race_faithful`)

| Quadrant | Available in corpus | Written | Distinct | Also under the opposite label | Train / val / test |
|---|---:|---:|---:|---:|---|
| pos_aa | 84,199 | 46,500 | 45,262 | 5,185 | 39,000 / 2,500 / 2,500 |
| pos_wh | 1,517,907 | 46,500 | 44,699 | 298 | 39,000 / 2,500 / 2,500 |
| neg_aa | 47,135 | 44,046 | 38,249 | 5,185 | 39,000 / 2,500 / 2,500 |
| neg_wh | 678,442 | 46,500 | 44,941 | 298 | 39,000 / 2,500 / 2,500 |

Test sentences also in training (paper split): 1,687. Mean tokens per tweet: pos_aa 10.7, pos_wh 11.17, neg_aa 10.68, neg_wh 11.38.

As in the original code, a tweet is written once per emoji it contains, so *Written* can exceed *Available* (which counts each tweet once), and tweets with both happy and sad emojis are written under both labels.

Where the training and test portions come from (top emojis, paper split):

| Quadrant | Training (first 41,500) | Test (next 2,500) |
|---|---|---|
| pos_aa | 😂 23,235, :) 5,265, :-) 3,505 | 😂 2,500 |
| pos_wh | :) 41,500 | :) 2,500 |
| neg_aa | 😩 13,486, 😒 12,116, 😔 3,727 | 😢 737, 😖 721, 😨 216 |
| neg_wh | :( 41,500 | :( 2,500 |

### Quirks of the original preprocessing, counted on the full corpus

High-confidence tweets (posterior > 0.8) with at least one happy or sad emoji.

| | AAE group | White-aligned group |
|---|---:|---:|
| emoji tweets | 122,938 | 2,250,944 |
| happy emojis only | 72,216 (58.7%) | 1,478,256 (65.7%) |
| sad emojis only | 38,739 (31.5%) | 733,037 (32.6%) |
| both happy and sad (enter both classes in the original) | 11,983 (9.7%) | 39,651 (1.8%) |
| two or more emojis of one class (counted twice in the original) | 11,093 (9.0%) | 137,011 (6.1%) |
| sad only through the sob emoji (never collected in the original) | 761 (0.6%) | 79,982 (3.6%) |

