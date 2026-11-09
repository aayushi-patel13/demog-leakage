#!/usr/bin/env bash
# Run the authors' original code (Elazar & Goldberg 2018; Python 2.7, DyNet) unchanged on the
# faithful TwitterAAE data, with the commands from the authors' runs.md, so its numbers can be set
# beside the PyTorch port's on exactly the same data.
#
#   bash scripts/run_original.sh
#   BASE_EPOCHS=8 ADV_EPOCHS=10 ATT_EPOCHS=100 bash scripts/run_original.sh   # the defaults
#
# Runs, on CPU:
#   1. sentiment-only encoder (--ro -1 --task sentiment --type 1)    } side by side,
#   2. adversarial encoder, lambda 1 (--ro 1.0 --task sent_race)      } one core each
#   3. attackers on the frozen encoders: the baseline's best model, and the adversarial run's best
#      model and epoch-10 checkpoint (the authors attack both in runs.md)
# On a 2-core machine a training epoch takes 20-35 minutes, so the defaults need about 7 hours.
# Logs and a summary table go to results/original_code/.
#
# If training was interrupted (e.g. the machine went to sleep), the checkpoints it saved are kept:
#   SKIP_DATA=1 SKIP_TRAIN=1 bash scripts/run_original.sh     # attackers + summary only
#
# Python 2.7: $PY27 if set, else Docker if available, else built from source into ~/py27
# (about 5 minutes, first time only).
set -euo pipefail
BASE_EPOCHS=${BASE_EPOCHS:-8}
ADV_EPOCHS=${ADV_EPOCHS:-10}
ATT_EPOCHS=${ATT_EPOCHS:-100}
ROOT=$(pwd)
ORIG=${ORIG:-$ROOT/original}
DATA=${DATA:-data/processed/sent_race_faithful}
OUT=$ROOT/results/original_code
mkdir -p "$OUT"

# 1. the original repository, untouched
if [ ! -d "$ORIG/.git" ]; then
  git clone -q https://github.com/yanaiela/demog-text-removal.git "$ORIG"
fi
echo "original code: github.com/yanaiela/demog-text-removal at $(git -C "$ORIG" log -1 --format='%h (%ad)' --date=short)" \
  | tee "$OUT/environment.txt"

# 2. our faithful data, in the original's file format (word ids + vocab, as its make_data.py writes)
if [ "${SKIP_DATA:-0}" != "1" ]; then
  python scripts/to_original_format.py "$DATA" "$ORIG/data/processed/sent_race" | tee -a "$OUT/environment.txt"
fi

# 3. Python 2.7 with mainline DyNet 2.1.2 and the original's other requirements
PKGS=(dyNET==2.1.2 numpy==1.16.6 scipy==1.2.3 scikit-learn==0.20.4)
build_py27() {   # CPython 2.7.18 from source; packages fetched by this Python 3's pip
  local P=$HOME/py27
  if [ ! -x "$P/bin/python2.7" ]; then
    echo "building Python 2.7.18 from source (first time only, about 5 minutes)..."
    local B; B=$(mktemp -d)
    git clone -q --depth 1 --branch v2.7.18 https://github.com/python/cpython.git "$B/cpython" 2> /dev/null
    (cd "$B/cpython" && ./configure -q --prefix="$P" --enable-unicode=ucs4 > /dev/null 2>&1 \
       && make -s -j"$(nproc)" > /dev/null 2>&1 && make -s install > /dev/null 2>&1)
    "$P/bin/python2.7" -m ensurepip > /dev/null 2>&1
    rm -rf "$B"
  fi
  if ! "$P/bin/python2.7" -c "import dynet, sklearn, docopt" > /dev/null 2>&1; then
    local W; W=$(mktemp -d)
    python -m pip download -q -d "$W" --python-version 27 --platform manylinux1_x86_64 \
        --implementation cp --abi cp27mu --only-binary=:all: "${PKGS[@]}"
    python -m pip download -q -d "$W" --no-deps --no-binary :all: docopt==0.6.2
    "$P/bin/python2.7" -m pip install -q --no-index --find-links "$W" "${PKGS[@]}" docopt==0.6.2 2> /dev/null
    rm -rf "$W"
  fi
  PY=("$P/bin/python2.7")
}
if [ -n "${PY27:-}" ]; then          # an existing Python 2.7 with those packages
  PY=("$PY27")
elif docker info > /dev/null 2>&1; then
  echo "building the Python 2.7 Docker image (first time only)..."
  printf 'FROM python:2.7.18-slim\nRUN pip install --no-cache-dir %s docopt==0.6.2\n' "${PKGS[*]}" \
      | docker build -q -t demog-py27 - > /dev/null
  PY=(docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT:$ROOT" -v "$ORIG:$ORIG" -w "$ROOT" demog-py27 python)
else
  build_py27
fi
"${PY[@]}" -c "import sys, dynet, sklearn; print('Python ' + sys.version.split()[0] + ', DyNet 2.1.2, scikit-learn ' + sklearn.__version__)" 2>/dev/null \
  | tee -a "$OUT/environment.txt"
echo "machine: $(nproc) CPU cores; started $(date -u '+%F %T') UTC" | tee -a "$OUT/environment.txt"

# 4. the original scripts, with the options from the authors' runs.md
COMMON=(--dynet-seed 12345 --dynet-autobatch 1 --dynet-mem 2048)
run() {   # run <log name> <script> <arguments...>
  local name=$1 script=$2; shift 2
  {
    echo "== $(date -u '+%F %T') UTC  python $script $*"
    "${PY[@]}" scripts/original_launcher.py "$ORIG/src/models/$script" "$@" 2>&1
    echo "== $(date -u '+%F %T') UTC  finished (exit $?)"
  } > "$OUT/$name.txt"
}
M="$ORIG/models"
mkdir -p "$M"
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
rm -rf "$M/sentiment-type:1-ro:-1" "$M/sent_race-n_adv:1-ro:1.0"

echo "training (baseline $BASE_EPOCHS epochs, adversarial $ADV_EPOCHS epochs, side by side)..."
run baseline_sentiment trainer.py "${COMMON[@]}" --ro -1 --task sentiment --type 1 --epochs "$BASE_EPOCHS" &
P1=$!
run adversarial_lambda1 trainer.py "${COMMON[@]}" --ro 1.0 --task sent_race --num_adv 1 --epochs "$ADV_EPOCHS" &
P2=$!
wait $P1 $P2
else
  echo "SKIP_TRAIN=1: keeping the trained models in $M; attacking them as they are"
fi
rm -rf "$M"/attacker-*

echo "attackers ($ATT_EPOCHS epochs each)..."
run attacker_baseline attacker.py "${COMMON[@]}" --ro -1 --task race \
    --model "sentiment-type:1-ro:-1/best_model" --epochs "$ATT_EPOCHS" &
P3=$!
run attacker_adversarial_best attacker.py "${COMMON[@]}" --ro 1.0 --task sent_race --num_adv 1 \
    --model "sent_race-n_adv:1-ro:1.0/best_model" --epochs "$ATT_EPOCHS" &
P4=$!
wait $P3 $P4
if [ -e "$M/sent_race-n_adv:1-ro:1.0/epoch_10" ]; then
  run attacker_adversarial_epoch10 attacker.py "${COMMON[@]}" --ro 1.0 --task sent_race --num_adv 1 \
      --model "sent_race-n_adv:1-ro:1.0/epoch_10" --epochs "$ATT_EPOCHS"
fi

echo "finished $(date -u '+%F %T') UTC" >> "$OUT/environment.txt"
python scripts/summarise_original.py "$OUT"
echo "ORIGINAL CODE RUNS FINISHED (logs and summary in $OUT)"
