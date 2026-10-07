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
| `scripts/run_original.sh` | runs the authors' original Python 2.7 / DyNet code on the faithful data (see below) |
| `scripts/finish_reddit.sh` | after collection: common end date, cleaning, labels, annotation sheet, summary, zip for Colab |
| `src/report_reddit.py` | summary of the Reddit dataset and the annotation scores (`results/reddit_data.md`) |
| `src/summarise_results.py` | every result in one table with 95% confidence intervals (`results/summary.md`) |
| `scripts/colab_run.py` | every training run on a Colab GPU, results saved to Google Drive, resumable |
| `scripts/smoke_test.sh` | runs everything on mock data in about 5 minutes |

## Environment

* **GitHub Codespaces** (2 cores, 8 GB): data download, preparation, Reddit
  collection and the smoke test. Install with
  `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`.
* **Google Colab** (T4 GPU): the training runs. Put `processed_data.zip`
  (made in Codespaces with
  `cd data/processed && zip -r ../../processed_data.zip sent_race sent_race_faithful`)
  and `reddit_data.zip` (made by `scripts/finish_reddit.sh`) in `MyDrive/comp8240`,
  then run in one Colab cell:

  ```
  from google.colab import drive; drive.mount('/content/drive')
  %cd /content
  !rm -rf demog-leakage && git clone -q https://github.com/aayushi-patel13/demog-leakage.git
  %cd /content/demog-leakage
  !python scripts/colab_run.py            # add --extra for the adversarial check
  ```

  The prepared tweets and comments are not in this repository.

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

`run-all` runs the paper's four balanced experiments (and `adv-check`, run by
`scripts/colab_run.py --extra`, repeats the adversarial setting with two more
random seeds and one 60-epoch run attacked after epochs 20, 40 and 60):

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

### The authors' original code

```bash
bash scripts/run_original.sh          # about 5 min setup, then roughly 25 min + 40 min on 2 CPU cores
```

This clones github.com/yanaiela/demog-text-removal, writes the faithful
TwitterAAE data in its file format (word ids plus a vocabulary file, as its
`make_data.py` produces), and runs its `trainer.py` unchanged under Python 2.7
with mainline DyNet 2.1.2: one epoch of the sentiment baseline and one of
adversarial training (lambda = 1), with the options from its `runs.md`.
Python 2.7 comes from Docker if available, otherwise it is built from source
into `~/py27`. The authors used their own DyNet fork, so
`scripts/original_launcher.py` applies two shims without editing their files:
the fork's `flip_gradient(x, ro)` becomes mainline `scale_gradient(x, -ro)`,
and the TensorBoard logger becomes a no-op. On CPU a training pass takes
about 25 minutes, so the original code is run as a check on the port, and the
full experiments use the port on a GPU. Logs: `results/original_code/`.

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
bash scripts/finish_pan17.sh        # download (53 MB), prepare, zip for Colab
# then on Colab: scripts/colab_run.py trains it as pan17_gb_us
```

Emoji-labelled tweets only (as in the original), retweets left out, texts
identical after normalisation removed, 20% of authors held out for test.

## Deviations from the original, and why

* **Framework.** The released code is Python 2 with a custom DyNet fork. It
  still runs (see *The authors' original code*), but only on CPU, so it is
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
* **Duplicates after normalisation** (`clean` mode only). The original removes
  tweets whose raw text repeats, but different raw tweets can become the same
  token sequence once mentions, links and emojis are normalised away; on the
  full corpus about 2% of test sentences also occurred in training, and some
  texts appeared under both labels. The clean data drops every such text.
* **Gender/age branch not reproduced.** PAN16 ships tweet ids only and the
  Twitter API that rehydrated them is no longer free.
* **Reddit collection cut short.** From 7 October 2026 the archive answered
  most requests with "Timeout. Maybe slow down a bit". By then r/Nigeria had
  reached 6 March 2025 and r/philadelphia only April 2024, so r/philadelphia is
  left out (its file moved to `data/reddit/raw/excluded/`) and cleaning runs with `--until 2025-03-06`: both groups cover
  1 January 2024 to 5 March 2025. The collector also skipped 364 one-hour
  windows that the archive kept refusing (188 US, 176 Nigerian; listed in the
  collection logs).
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
