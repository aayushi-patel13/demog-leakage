#!/usr/bin/env bash
# Download the PAN 2017 author profiling training set (Rangel et al., 2017) from Zenodo.
set -euo pipefail
DEST="data/raw/pan17"
mkdir -p "$DEST"
cd "$DEST"
URL="https://zenodo.org/records/3745980/files/pan17-author-profiling-training-dataset-2017-03-10.zip?download=1"
wget -c "$URL" -O pan17-training.zip
unzip -o pan17-training.zip >/dev/null
# the English part may be nested one level down; find it
EN=$(find . -type d -name en | head -1)
echo "English folder: $EN"
ls "$EN" | head -5
echo "authors: $(ls "$EN"/*.xml 2>/dev/null | wc -l)"
head -3 "$EN/truth.txt" || true
echo
echo "Next: python src/prepare_pan17.py data/raw/pan17/${EN#./} data/processed/pan17_gb_us"
