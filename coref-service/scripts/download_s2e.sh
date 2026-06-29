#!/usr/bin/env bash
#
# Download the trained s2e-coref checkpoint used by cascade mode (~1.6 GB).
# Idempotent: skips the download if the checkpoint is already present.
#
# Usage:  bash scripts/download_s2e.sh [DEST_DIR]
# Default DEST_DIR is ./s2e-model (the path docker-compose mounts at /models/s2e).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/../s2e-model}"
URL="https://www.dropbox.com/sh/7hpw662xylbmi5o/AAC3nfP4xdGAkf0UkFGzAbrja?dl=1"

mkdir -p "$DEST"

if [ -f "$DEST/pytorch_model.bin" ]; then
  echo "✓ s2e checkpoint already present in $DEST — nothing to do."
  exit 0
fi

echo "Downloading s2e-coref checkpoint (~1.6 GB) into $DEST ..."
tmp="$(mktemp -d)/s2e.zip"
curl -L --fail --retry 5 --retry-delay 5 -o "$tmp" "$URL"

echo "Extracting ..."
unzip -o -q "$tmp" -d "$DEST"
rm -f "$tmp"

# The archive may nest the files one folder deep — flatten if so.
if [ ! -f "$DEST/pytorch_model.bin" ]; then
  inner="$(find "$DEST" -name pytorch_model.bin -print -quit || true)"
  if [ -n "${inner:-}" ]; then
    mv "$(dirname "$inner")"/* "$DEST"/ 2>/dev/null || true
  fi
fi

if [ -f "$DEST/pytorch_model.bin" ]; then
  echo "✓ Done. Checkpoint ready in $DEST"
else
  echo "✗ Expected pytorch_model.bin in $DEST but it is missing." >&2
  echo "  Check the download URL or unzip output." >&2
  exit 1
fi
