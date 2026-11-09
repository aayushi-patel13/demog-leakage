#!/usr/bin/env bash
# Finish the original-code comparison in one short, clean run. A Codespace that sleeps stops long
# jobs, so the attacker on the adversarial model runs for ATT_EP epochs (default 30) and the summary
# compares every attacker over the same number of epochs. The script first stops anything left over
# from earlier attempts and moves their partial attacker logs to results/original_code/interrupted/.
#   nohup bash scripts/resume_original.sh > resume.log 2>&1 &
set -uo pipefail
ROOT=$(pwd)
OUT=$ROOT/results/original_code
M=$ROOT/original/models
ATT_EP=${ATT_EP:-30}
LOG=$OUT/attacker_adversarial_best.txt

# 1. stop earlier copies of this job and their children (never this shell itself)
pgrep -f "resume_original[.]sh|port_epoch_match[.]sh|src/experiments[.]py|original_launcher[.]py" > /tmp/old_pids.txt || true
while read -r p; do [ "$p" != "$$" ] && kill "$p" 2> /dev/null && echo "stopped old process $p"; done < /tmp/old_pids.txt

# 2. Docker can take a minute or two to come up after a restart
for _ in $(seq 1 60); do docker info > /dev/null 2>&1 && break; sleep 5; done
if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running; reload the Codespace window and run this script again"; exit 1
fi
docker ps -q | xargs -r docker kill > /dev/null
docker image inspect demog-py27 > /dev/null 2>&1 || \
  printf 'FROM python:2.7.18-slim\nRUN pip install --no-cache-dir dyNET==2.1.2 numpy==1.16.6 scipy==1.2.3 scikit-learn==0.20.4 docopt==0.6.2\n' \
  | docker build -q -t demog-py27 - > /dev/null

# 3. keep earlier partial attacker logs for the record, out of the way of the summary
mkdir -p "$OUT/interrupted"
for f in "$LOG" "$OUT/attacker_adversarial_best_interrupted.txt"; do
  [ -f "$f" ] && mv "$f" "$OUT/interrupted/$(basename "$f" .txt)_$(date -u +%Y%m%d%H%M).txt"
done

# 4. attacker on the adversarial model, ATT_EP epochs, beside the remaining port runs
rm -rf "$M/attacker-sent_race-n_adv:1-ro:1.0"
(
  echo "== $(date -u '+%F %T') UTC  python attacker.py, $ATT_EP epochs, on sent_race-n_adv:1-ro:1.0/best_model"
  docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT:$ROOT" -w "$ROOT" demog-py27 python \
    scripts/original_launcher.py "$ROOT/original/src/models/attacker.py" \
    --dynet-seed 12345 --dynet-autobatch 1 --dynet-mem 2048 --ro 1.0 --task sent_race --num_adv 1 \
    --model "sent_race-n_adv:1-ro:1.0/best_model" --epochs "$ATT_EP" 2>&1
  echo "== $(date -u '+%F %T') UTC  finished (exit $?)"
) > "$LOG" &
P1=$!
OMP_NUM_THREADS=1 bash scripts/port_epoch_match.sh > "$ROOT/port_epoch_match.log" 2>&1
wait "$P1"
python scripts/summarise_original.py "$OUT"
echo "ALL DONE"
