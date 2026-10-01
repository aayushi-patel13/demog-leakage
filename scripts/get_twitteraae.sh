#!/usr/bin/env bash
# Download the TwitterAAE corpus (Blodgett et al., 2016), about 5.5 GB.
# It is read directly from the zip, never unpacked. The download can resume.
set -euo pipefail
URL="http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip"
DEST="data/raw"
mkdir -p "$DEST"
cd "$DEST"

echo "Free space here:"; df -h . | tail -1
SIZE=$(curl -sIL "$URL" | awk 'tolower($1)=="content-length:"{print $2}' | tail -1 | tr -d '\r')
if [ -n "${SIZE:-}" ]; then
  echo "Download size: $((SIZE / 1024 / 1024)) MB"
fi

wget -c "$URL" -O TwitterAAE-full-v1.zip
# No unzipping: the zip is ~5.5 GB and unpacks to far more than a Codespace's
# disk. prepare_twitteraae.py reads the files straight out of the zip.
echo "Files inside the zip:"
unzip -l TwitterAAE-full-v1.zip
echo
echo "Next: python src/prepare_twitteraae.py data/raw/TwitterAAE-full-v1.zip data/processed/sent_race"
