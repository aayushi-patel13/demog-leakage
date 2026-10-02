# Adversarial removal of demographic attributes: replication and extension

COMP8240 project, Aayushi Dharmeshbhai Patel (49136186), Macquarie University.

Reproduces **Elazar & Goldberg (2018), "Adversarial Removal of Demographic
Attributes from Text Data" (EMNLP)** and extends it to new data. The paper's
claim: adversarial training drives the *online* adversary to chance, yet a fresh
*post-hoc attacker* trained on the frozen representation still recovers the
demographic attribute well above chance.

## What is here

| Path | What it does |
|---|---|
| `src/prepare_twitteraae.py` | builds the paper's balanced sentiment/race data (up to 166k train / 10k test) from TwitterAAE, streaming, in 8 GB RAM |
| `src/report_twitteraae.py` | summary of the prepared data (sizes, repeats, label conflicts, quirk counts) for the write-up |
| `src/models.py`, `src/optim.py` | PyTorch port of the original DyNet model: 300-d embeddings, 300-d LSTM, 300-d tanh heads, gradient reversal, DyNet-style momentum SGD |
| `src/experiments.py` | trains encoders, runs post-hoc attackers, writes a results table next to the paper's numbers |
| `src/collect_reddit.py` | collects comments from US and Nigerian subreddits through the Arctic Shift archive API |
| `src/preprocess_reddit.py` | cleaning, bot and cross-group author removal, author caps, dedup, Pidgin markers, topic masking |
| `src/label_reddit.py` | distant-supervision sentiment (emoji, optionally VADER), length-matched balanced quadrants, author-disjoint split |
| `src/annotation.py` | blind human-annotation sample (200 comments) and agreement scoring |
| `src/prepare_pan17.py` | existing dataset: PAN 2017 English, variety (GB vs US) or gender as the protected attribute |
| `notebooks/replicate_on_colab.ipynb` | the replication on a free Colab GPU |
| `scripts/smoke_test.sh` | runs everything on mock data in about 5 minutes |

## Environment

* **GitHub Codespaces** (the unit's VM): open the repo, *Code → Codespaces →
  Create*. The dev container installs Python 3.11, CPU PyTorch and the
  requirements. 2 cores / 8 GB is enough for everything; the replication
  training takes about 3-4 hours on CPU.
* **Google Colab** (the unit's suggested notebook environment): use
  `notebooks/replicate_on_colab.ipynb` with a T4 GPU for the training step.
  Prepare the data in Codespaces, then
  `cd data/processed && zip -r ../../processed_data.zip sent_race sent_race_faithful`,
  download `processed_data.zip` and put it in `MyDrive/comp8240/` on Google Drive.
  (The prepared tweets are not committed to the public repository.)

First, check the environment (about 5 minutes):

```bash
bash scripts/smoke_test.sh          # must end with SMOKE TEST PASSED
```

## 1. Replication on the original data (TwitterAAE, race branch)

```bash
bash scripts/get_twitteraae.sh                   # ~5.5 GB zip, read directly (no unzipping)
python src/prepare_twitteraae.py data/raw/TwitterAAE-full-v1.zip data/processed/sent_race
python src/experiments.py run-all --data data/processed/sent_race --epochs 20 --tag twitteraae
cat results/twitteraae/results.md
```

`run-all` runs the paper's four balanced experiments:

| | Experiment | Paper |
|---|---|---|
| E1 | encoder trained on sentiment only: sentiment accuracy | 67.4 |
| E2 | encoder trained on race only: race accuracy | 83.9 |
| E3 | post-hoc attacker on the E1 encoder: leakage, no defence | 64.5 |
| E4 | adversarial training, lambda = 1: sentiment / leakage / delta | 64.7 / 56.0 / 5.0 |

`python src/report_twitteraae.py` summarises both prepared datasets in
`results/twitteraae_data.md`: tweets available and written per quadrant,
repeats, tweets under both labels, the split sizes used, and the quirk counts.
If a quadrant is short of the paper's 44,000 tweets, all four quadrants get the
same smaller sizes (test and validation 2,500 each, training the rest), so the
data stays balanced and the attacker's chance level stays at 50%.

Both modes write `data_report.json`, which counts how many tweets each quirk
of the original preprocessing affects (see *Deviations*), plus emoji
frequencies per dialect group. `--mode faithful` builds the dataset the way the
original code behaves as written, for an ablation against the clean data:

```bash
python src/prepare_twitteraae.py data/raw/TwitterAAE-full-v1.zip data/processed/sent_race_faithful --mode faithful
```

## 2. Constructed dataset: US vs Nigerian English on Reddit

```bash
python src/collect_reddit.py                  # a few hours at 1 request/s; resumable
python src/preprocess_reddit.py               # -> data/reddit/clean/
python src/label_reddit.py                    # -> data/reddit/processed/{tokens,masked}/
python src/annotation.py sample --n 50        # -> annotation/sample_for_annotation.csv (blind)
# annotate the 200 comments (human_sentiment: pos/neg/neutral, human_variety: NG/US/unsure),
# save as annotation/sample_annotated.csv, then:
python src/annotation.py score annotation/sample_annotated.csv
python src/experiments.py run-all --data data/reddit/processed/tokens --epochs 20 \
    --tag reddit_tokens --attribute "variety (NG vs US)" --no-paper
python src/experiments.py run-all --data data/reddit/processed/masked --epochs 20 \
    --tag reddit_masked --attribute "variety (NG vs US)" --no-paper
```

Sampling: the US subreddits are busy (thousands of comments a day in the
largest), so each contributes up to 100 comments per day, taken from a random
start time each day, which spreads the sample over all 18 months and all
hours of the day. Taking the first comments in time order instead would put
the whole US sample in the first days of January 2024 while the Nigerian
sample spans the full window, a time and topic confound. The Nigerian
subreddits are far smaller and are collected in full.

The archive answers HTTP 422 when a query is too expensive (large subreddits
over a long time range); the collector then narrows its time window and
continues. Any window it still has to skip is listed in
`data/reddit/raw/skipped_windows.jsonl`, so gaps in the data can be reported.

Subreddits and the time window are in `config/reddit_groups.json`. The
`masked` variant replaces place names, politicians, currencies and similar
topic giveaways with `_TOPIC_`, to test whether an attacker is reading dialect
or merely topic. `label_reddit.py --labels emoji` restricts to emoji labels
(closest to the paper); the default adds strong VADER labels because emojis
are rarer on Reddit.

## 3. Existing dataset: PAN 2017 author profiling

```bash
bash scripts/get_pan17.sh
python src/prepare_pan17.py data/raw/pan17/<path>/en data/processed/pan17_gb_us
python src/experiments.py run-all --data data/processed/pan17_gb_us --epochs 20 \
    --tag pan17_gb_us --attribute "variety (GB vs US)" --no-paper
```

## Deviations from the original, and why

* **Framework.** The released code is Python 2 with a custom DyNet fork; it is
  reimplemented in PyTorch with the same sizes and optimiser settings (momentum
  SGD, lr 0.01, summed losses over batches of 32, gradient clipping at 5,
  sparse embedding updates, dropout 0.2).
* **Epochs.** 20 instead of 100, for the compute available; every epoch is
  logged so convergence can be checked.
* **Model selection.** The original keeps the epoch with the best *test*
  accuracy, for both the encoder and the attacker. We report that ("paper
  protocol") and, alongside, the test score at the epoch chosen on a separate
  validation set ("held-out selection").
* **Preprocessing quirks in the original** (kept in `--mode faithful`, fixed
  in the default `clean` mode): tweets are collected emoji by emoji, so one with
  two different happy emojis is counted twice; the check meant to drop tweets
  with both happy and sad emojis compares tokens with regex strings and never
  fires, so such tweets enter both classes; the pattern for the sob emoji has a
  missing backslash, so under Python 2 it never matches and sob-only tweets are
  never collected as sad; and quadrants are ordered by emoji rather than
  shuffled, so train and test can come from different emojis.
* **Gender/age branch not reproduced.** PAN16 ships tweet ids only and the
  Twitter API that rehydrated them is no longer free.
* **Reddit access.** The proposal planned the official Reddit API; self-service
  keys were withdrawn in late 2025, so comments come from the Arctic Shift
  public archive instead.

## Data and ethics

Reddit authors are stored only as salted hashes; usernames are never written.
Raw downloads (`data/raw/`, `data/reddit/raw/`) are not committed. The dialect
labels in every dataset are coarse proxies (inferred from geography or from the
subreddit), not self-reported identity, and results should be read that way.

## References

Elazar & Goldberg (2018), EMNLP. Blodgett, Green & O'Connor (2016), EMNLP.
Ganin & Lempitsky (2015), ICML. Rangel et al. (2017), PAN at CLEF.
Hutto & Gilbert (2014), VADER, ICWSM.
