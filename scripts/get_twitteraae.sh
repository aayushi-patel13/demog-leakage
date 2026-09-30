#!/usr/bin/env bash
# Download and unzip the TwitterAAE corpus (Blodgett et al., 2016).
# The file is large; the script checks free disk space first and can resume.
set -euo pipefail
URL="http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip"
DEST="data/raw"
mkdir -p "$DEST"
cd "$DEST"

echo "Free space here:"; df -h . | tail -1
SIZE=$(curl -sIL "$URL" | awk 'tolower($1)=="content-length:"{print $2}' | tail -1 | tr -d '\r')
if [ -n "${SIZE:-}" ]; then
  echo "Download size: $((SIZE / 1024 / 1024)) MB (unzipped it is several times larger)"
fi

wget -c "$URL" -O TwitterAAE-full-v1.zip
echo "Unzipping..."
unzip -o TwitterAAE-full-v1.zip -d TwitterAAE-full-v1 >/dev/null
echo "Contents:"
ls -lh TwitterAAE-full-v1 | head -20
for f in TwitterAAE-full-v1/*; do
  [ -f "$f" ] && { echo "--- first line of $f"; head -1 "$f" | cut -c1-300; }
done
echo
echo "Next: python src/prepare_twitteraae.py data/raw/TwitterAAE-full-v1 data/processed/sent_race"
