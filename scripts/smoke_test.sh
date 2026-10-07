#!/usr/bin/env bash
# Five-minute check that the whole pipeline runs in your environment, on mock
# data in the real file formats. Run this before the big downloads.
# The numbers it prints are meaningless; only "SMOKE TEST PASSED" matters.
set -euo pipefail
python tests/make_mock_twitteraae.py --rows 60000 --out data/mock/TwitterAAE-full-v1
(cd data/mock && rm -f TwitterAAE-full-v1.zip && python -m zipfile -c TwitterAAE-full-v1.zip TwitterAAE-full-v1)
python src/prepare_twitteraae.py data/mock/TwitterAAE-full-v1.zip data/mock/sent_race > /dev/null
python src/prepare_twitteraae.py data/mock/TwitterAAE-full-v1.zip data/mock/sent_race_faithful --mode faithful > /dev/null
python src/report_twitteraae.py --clean data/mock/sent_race --faithful data/mock/sent_race_faithful \
  --out data/mock/twitteraae_data.md > /dev/null
python - <<'EOF'
# clean data: one tweet per line, nothing repeated, nothing under both labels
import json, sys
sys.path.insert(0, "src")
from data import read_lines
rep = json.load(open("data/mock/sent_race/data_report.json"))
lines = {q: read_lines(f"data/mock/sent_race/{q}.txt") for q in rep["quadrants"]}
for q, v in rep["quadrants"].items():
    assert len(lines[q]) == v["written"], (q, len(lines[q]), v["written"])
    with open(f"data/mock/sent_race/{q}.txt", encoding="utf-8") as fh:
        assert len(fh.read().splitlines()) == v["written"], f"{q}: a tweet spans two lines"
allx = [l for v in lines.values() for l in v]
assert len(allx) == len(set(allx)), "clean data has repeated texts"
assert rep["test_sentences_also_in_train"] == 0
print("clean data checks OK")
EOF
python tests/mock_pan17.py data/mock/pan17/en > /dev/null
python src/prepare_pan17.py data/mock/pan17/en data/mock/pan17_gb_us > /dev/null
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
