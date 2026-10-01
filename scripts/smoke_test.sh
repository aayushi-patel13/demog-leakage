#!/usr/bin/env bash
# Five-minute check that the whole pipeline runs in your environment, on mock
# data in the real file formats. Run this before the big downloads.
# The numbers it prints are meaningless; only "SMOKE TEST PASSED" matters.
set -euo pipefail
python tests/make_mock_twitteraae.py --rows 60000 --out data/mock/TwitterAAE-full-v1
(cd data/mock && rm -f TwitterAAE-full-v1.zip && python -m zipfile -c TwitterAAE-full-v1.zip TwitterAAE-full-v1)
python src/prepare_twitteraae.py data/mock/TwitterAAE-full-v1.zip data/mock/sent_race > /dev/null
python src/prepare_twitteraae.py data/mock/TwitterAAE-full-v1.zip data/mock/sent_race_faithful --mode faithful > /dev/null
python tests/mock_reddit.py
python src/preprocess_reddit.py --raw data/mock/reddit_raw --out data/mock/reddit_clean > /dev/null
python src/label_reddit.py --clean data/mock/reddit_clean/comments.jsonl --out data/mock/reddit_processed > /dev/null
python src/annotation.py sample --clean data/mock/reddit_clean/comments.jsonl \
  --processed data/mock/reddit_processed/tokens --out data/mock/annotation --n 5
python src/experiments.py run-all --data data/mock/sent_race --epochs 1 --attacker-epochs 2 \
  --tag smoke_twitteraae --max-train 2000 > /dev/null
python src/experiments.py run-all --data data/mock/reddit_processed/tokens --epochs 1 --attacker-epochs 2 \
  --tag smoke_reddit --max-train 2000 --attribute "variety (NG vs US)" --no-paper > /dev/null
test -f results/smoke_twitteraae/results.md && test -f results/smoke_reddit/results.md
rm -rf models/smoke_* results/smoke_*
echo "SMOKE TEST PASSED"
