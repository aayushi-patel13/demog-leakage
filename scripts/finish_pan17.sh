#!/usr/bin/env bash
# PAN 2017 (GB vs US English): download, prepare, zip for Colab. About 2 minutes.
set -euo pipefail
bash scripts/get_pan17.sh
EN=$(find data/raw/pan17 -type d -name en -print -quit)
python src/prepare_pan17.py "$EN" data/processed/pan17_gb_us
rm -f pan17_data.zip
(cd data/processed && zip -qr ../../pan17_data.zip pan17_gb_us)
echo "packed -> pan17_data.zip ($(du -h pan17_data.zip | cut -f1)) for Colab"
