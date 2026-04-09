#!/usr/bin/env bash
set -u
set -o pipefail

EXE="./build/qr_comparer"
IMG_DIR="data/img"
JSON_DIR="out/json"
OUT_MASKED_DIR="out/masked"
OUT_CSV="out/summary.csv"

mkdir -p "$OUT_MASKED_DIR" "$(dirname "$OUT_CSV")"

echo "file_name,has_json,exit_code,full,masked,full_text,masked_text" > "$OUT_CSV"

find "$IMG_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -print0 |
while IFS= read -r -d '' img; do
  base="$(basename "$img")"
  stem="${base%.*}"

  json="$JSON_DIR/$stem.json"
  masked="$OUT_MASKED_DIR/$stem.png"
  log="$OUT_MASKED_DIR/$stem.txt"

  has_json="YES"
  if [[ ! -f "$json" ]]; then
    has_json="NO"
    echo "[SKIP] $base (no json: $json)"
    echo "${base},${has_json},,SKIP,SKIP,\"\",\"\"" >> "$OUT_CSV"
    continue
  fi

  echo "[RUN] $base"
  "$EXE" "$img" "$json" "$masked" > "$log" 2>&1
  code=$?

  full=$(grep -E '^FULL:' "$log"   | awk '{print $2}' | head -n1)
  masked_res=$(grep -E '^MASKED:' "$log" | awk '{print $2}' | head -n1)

  full_text=$(grep -E '^FULL_TEXT:' "$log" | sed 's/^FULL_TEXT: //' | head -n1 || true)
  masked_text=$(grep -E '^MASKED_TEXT:' "$log" | sed 's/^MASKED_TEXT: //' | head -n1 || true)

  full=${full:-NA}
  masked_res=${masked_res:-NA}

  full_text=${full_text//\"/\"\"}
  masked_text=${masked_text//\"/\"\"}

  echo "${base},${has_json},${code},${full},${masked_res},\"${full_text}\",\"${masked_text}\"" >> "$OUT_CSV"
done

echo "Done."
echo "Masked images: $OUT_MASKED_DIR"
echo "Summary CSV  : $OUT_CSV"
