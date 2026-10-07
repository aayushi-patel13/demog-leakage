#!/usr/bin/env bash
# Download the PAN 2017 author profiling training set (Rangel et al., 2017) from Zenodo.
# (No "cmd | head" pipes here: under pipefail, head closing the pipe early counts as a failure.)
set -euo pipefail
DEST="data/raw/pan17"
mkdir -p "$DEST"
cd "$DEST"
URL="https://zenodo.org/records/3745980/files/pan17-author-profiling-training-dataset-2017-03-10.zip?download=1"
if [ ! -s pan17-training.zip ]; then
  wget -c "$URL" -O pan17-training.zip
fi
unzip -oq pan17-training.zip
EN=$(find . -type d -name en -print -quit)
echo "English folder: $EN"
echo "authors: $(find "$EN" -name '*.xml' | wc -l)"
sed -n 1,3p "$EN/truth.txt"
