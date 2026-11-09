#!/usr/bin/env bash
# Train the PyTorch port for exactly as many epochs as the original-code run completed, on the same
# faithful data, attack it the same way (best test epoch, 100-epoch attacker), and add an
# epoch-matched comparison to results/original_code/summary.md. A part that already finished
# (its attack_z_enc-test.json exists) is skipped, so the script can be re-run after an interruption.
#   bash scripts/port_epoch_match.sh            # epochs read from the original-code logs
#   BASE_EP=2 ADV_EP=1 bash scripts/port_epoch_match.sh
set -euo pipefail
D=${DATA:-data/processed/sent_race_faithful}
R=results/original_code
O=models/epoch_match
BASE_EP=${BASE_EP:-$(grep -c "dev-task epoch" "$R/baseline_sentiment.txt" || true)}
ADV_EP=${ADV_EP:-$(grep -c "^dev, " "$R/adversarial_lambda1.txt" || true)}
echo "port, epoch-matched: sentiment-only $BASE_EP epoch(s), adversarial $ADV_EP epoch(s)"
if [ -f "$O/sentiment/attack_z_enc-test.json" ]; then
  echo "sentiment-only part already done"
else
  rm -rf "$O/sentiment"
  python src/experiments.py train --data "$D" --target y --epochs "$BASE_EP" --out "$O/sentiment" > "$R/port_epoch_match_sentiment.txt" 2>&1
  python src/experiments.py attack --data "$D" --model "$O/sentiment" --which test >> "$R/port_epoch_match_sentiment.txt" 2>&1
fi
if [ -f "$O/adversarial/attack_z_enc-test.json" ]; then
  echo "adversarial part already done"
else
  rm -rf "$O/adversarial"
  python src/experiments.py train --data "$D" --target y --lambd 1.0 --epochs "$ADV_EP" --out "$O/adversarial" > "$R/port_epoch_match_adversarial.txt" 2>&1
  python src/experiments.py attack --data "$D" --model "$O/adversarial" --which test >> "$R/port_epoch_match_adversarial.txt" 2>&1
fi
python scripts/summarise_original.py "$R"
echo "EPOCH-MATCHED PORT RUNS FINISHED"
