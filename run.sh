#!/usr/bin/env bash
set -euo pipefail

BIN="./build/qr_localizer"
IMG_DIR="./data/img"

OUT_JSON_DIR="./out/json"
OUT_VIS_DIR="./out/vis"

mkdir -p "$OUT_JSON_DIR" "$OUT_VIS_DIR"

shopt -s nullglob
for img in "$IMG_DIR"/*.jpg "$IMG_DIR"/*.png "$IMG_DIR"/*.jpeg; do
  base="$(basename "$img")"
  stem="${base%.*}"

  json_out="$OUT_JSON_DIR/$stem.json"
  vis_out="$OUT_VIS_DIR/$stem.png"

  echo "Processing: $base"
  "$BIN" "$img" --out "$json_out" --vis "$vis_out"
done

echo "Done."
echo "JSON dir: $OUT_JSON_DIR"
echo "IMG dir:  $OUT_VIS_DIR"
