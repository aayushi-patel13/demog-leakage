## Reddit: US vs Nigerian English

Comments created before 2025-03-06 (2024-01-01 onwards); both groups cover the same period.

| Subreddit | Collected | Covers | Kept after cleaning |
|---|---:|---|---:|
| r/Abuja (NG) | 52 | 2024-02-27 to 2025-06-14 | 22 |
| r/Nigeria (NG) | 235,620 | 2024-01-01 to 2025-03-06 | 79,257 |
| r/lagos (NG) | 665 | 2024-01-02 to 2025-06-30 | 348 |
| r/AskAnAmerican (US) | 54,700 | 2024-01-01 to 2025-06-30 | 24,853 |
| r/Atlanta (US) | 54,605 | 2024-01-01 to 2025-06-30 | 23,125 |
| r/chicago (US) | 54,700 | 2024-01-01 to 2025-06-30 | 26,977 |
| r/houston (US) | 54,700 | 2024-01-01 to 2025-06-30 | 24,278 |

Removed during cleaning:

* length outside 3 40: 125,815
* over author cap: 74,097
* after until date: 47,466
* deleted or removed: 18,759
* bot or deleted author: 4,548
* duplicate text: 3,870
* author in both groups: 1,627

| Group | Comments | Authors | Median tokens | With a sentiment emoji | 2+ Pidgin markers |
|---|---:|---:|---:|---:|---:|
| NG | 79,627 | 17,641 | 14 | 6,198 (7.8%) | 502 (0.6%) |
| US | 99,233 | 41,721 | 16 | 1,887 (1.9%) | 3 (0.0%) |

One-hour windows the archive refused even after retries (skipped): r/AskAnAmerican 1, r/Atlanta 141, r/Nigeria 176, r/chicago 2, r/houston 15.

Sentiment labels (before balancing): NG emoji 5,955, NG vader 15,568, US emoji 1,862, US vader 20,740; comments with both happy and sad emojis dropped: 268.

| Quadrant | Train | Test (held-out authors) |
|---|---:|---:|
| pos_ng | 5,030 | 555 |
| pos_us | 5,030 | 555 |
| neg_ng | 5,030 | 555 |
| neg_us | 5,030 | 555 |

Total 22,340 comments, balanced across the four quadrants and matched on length (3-5, 6-10, 11-20, 21-40 tokens) and on label source (each quadrant: 505 emoji, 5,080 vader).

### LLM judge (Claude, zero-shot, text only, blind to labels; see annotation/LLM_JUDGE.md)

* Distant sentiment label matches the LLM label in 60.0% of 200 comments (Cohen's kappa 0.375); the LLM called 28.0% neutral. By source: vader 61.1% (n=185), emoji 46.7% (n=15).
* LLM guess of NG vs US from the text alone: 95.5% correct when decided, 45.0% unsure; 75.0% counting unsure as a coin flip (comparable to an attacker's accuracy, chance = 50%).

