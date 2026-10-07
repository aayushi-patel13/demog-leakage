#!/usr/bin/env bash
# Final Reddit processing, run once collection has stopped:
#   common end date -> cleaning -> labels -> annotation sheet -> summary -> zip for Colab
# Usage: bash scripts/finish_reddit.sh            (end date = last day reached by r/Nigeria)
#        bash scripts/finish_reddit.sh 2025-02-01 (or give the end date yourself)
set -euo pipefail
RAW=${RAW:-data/reddit/raw}
CLEAN=${CLEAN:-data/reddit/clean}
PROC=${PROC:-data/reddit/processed}
ANN=${ANN:-annotation}
REF=${REF:-$RAW/ng_Nigeria.jsonl}

if pgrep -f collect_reddit.py > /dev/null; then
  echo "A collector is still running. Wait for it to finish, or stop it with: pkill -f collect_reddit.py"
  exit 1
fi

UNTIL=${1:-$(python - "$REF" <<'EOF'
import json, sys, datetime as dt
last = None
for ln in open(sys.argv[1], encoding="utf-8"):
    try:
        last = int(json.loads(ln)["created_utc"])
    except Exception:
        pass
print(dt.datetime.fromtimestamp(last, dt.timezone.utc).strftime("%Y-%m-%d"))
EOF
)}
echo "Common end date: comments before $UNTIL are kept (the last, partly collected day is left out)"

python src/preprocess_reddit.py --raw "$RAW" --out "$CLEAN" --until "$UNTIL" > /dev/null
echo "cleaned   -> $CLEAN/clean_report.json"
python src/label_reddit.py --clean "$CLEAN/comments.jsonl" --out "$PROC" > /dev/null
echo "labelled  -> $PROC/label_report.json"
python src/annotation.py sample --clean "$CLEAN/comments.jsonl" --processed "$PROC/tokens" --out "$ANN" --n 50
python src/report_reddit.py --clean "$CLEAN" --processed "$PROC" --annotation "$ANN" --logs data/reddit
rm -f reddit_data.zip && zip -qr reddit_data.zip "$PROC"
echo "packed    -> reddit_data.zip ($(du -h reddit_data.zip | cut -f1)) for Colab"
