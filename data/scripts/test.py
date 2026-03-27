import argparse
import csv
import itertools
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


def load_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_localizer(exe: Path, img_path: Path, timeout=20.0):
    p = subprocess.run(
        [str(exe), str(img_path)], capture_output=True, text=True, timeout=timeout
    )
    out = (p.stdout or "").strip().splitlines()
    if not out:
        return False, None

    head = out[0].strip().upper()
    if "NOT" in head:
        return False, None
    if "FOUND" not in head:
        return False, None
    if len(out) < 5:
        return False, None

    pts = []
    for i in range(1, 5):
        parts = out[i].strip().split()
        if len(parts) < 2:
            return False, None
        pts.append((float(parts[0]), float(parts[1])))

    return True, pts


def corner_errors_px_list(gt4, pred4):
    errs = []
    for g, p in zip(gt4, pred4):
        errs.append(math.hypot(float(p[0] - g[0]), float(p[1] - g[1])))
    return float(np.mean(errs)), float(np.max(errs))


def best_match_all_perms(gt4, pred4):
    best_mean = None
    best_max = None
    best_perm = None
    best_pred = None

    for perm in itertools.permutations(range(4)):
        cand = [pred4[i] for i in perm]
        mean_e, max_e = corner_errors_px_list(gt4, cand)
        if best_mean is None or mean_e < best_mean:
            best_mean = mean_e
            best_max = max_e
            best_perm = perm
            best_pred = cand

    return best_mean, best_max, str(best_perm), best_pred


def polygon_iou_convex(quad1, quad2):
    if quad1 is None or quad2 is None:
        return None
    p1 = np.array(quad1, dtype=np.float32).reshape((-1, 1, 2))
    p2 = np.array(quad2, dtype=np.float32).reshape((-1, 1, 2))

    area1 = abs(cv2.contourArea(p1))
    area2 = abs(cv2.contourArea(p2))
    if area1 <= 1e-6 or area2 <= 1e-6:
        return None

    inter_area, _ = cv2.intersectConvexConvex(p1, p2)
    union = area1 + area2 - inter_area
    if union <= 1e-6:
        return None
    return float(inter_area / union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.json (GT)")
    ap.add_argument("--img_dir", required=True, help="Directory with images")
    ap.add_argument("--exe", required=True, help="Path to localizer executable")
    ap.add_argument("--out", default="results.csv", help="Output CSV path")
    ap.add_argument(
        "--timeout", type=float, default=20.0, help="Timeout per image (sec)"
    )
    args = ap.parse_args()

    data = load_data(Path(args.data))
    items = data.get("images", [])
    if not items:
        raise SystemExit("No images in data.json under key 'images'")

    img_dir = Path(args.img_dir)
    exe = Path(args.exe)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "file_name",
        "width",
        "height",
        "algo_found",
        "iou",
        "corner_err_mean_px",
        "corner_err_max_px",
        "corner_match_mode",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in items:
            image_id = item.get("id", "")
            file_name = item.get("file_name", "")
            width = item.get("width", "")
            height = item.get("height", "")

            ann_list = item.get("annotations") or []
            if not ann_list or "corners" not in ann_list[0]:
                continue

            cgt = ann_list[0]["corners"]
            gt4 = [
                tuple(cgt["tl"]),
                tuple(cgt["tr"]),
                tuple(cgt["br"]),
                tuple(cgt["bl"]),
            ]

            img_path = img_dir / file_name
            found, pred4 = run_localizer(exe, img_path, timeout=args.timeout)

            row = {
                "image_id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "algo_found": "YES" if found else "NO",
                "iou": "",
                "corner_err_mean_px": "",
                "corner_err_max_px": "",
                "corner_match_mode": "",
            }

            if found and pred4 is not None:
                mean_e, max_e, mode, pred_best = best_match_all_perms(gt4, pred4)
                iou = polygon_iou_convex(gt4, pred_best)

                row["iou"] = f"{iou:.6f}" if iou is not None else ""
                row["corner_err_mean_px"] = f"{mean_e:.3f}"
                row["corner_err_max_px"] = f"{max_e:.3f}"
                row["corner_match_mode"] = mode

            writer.writerow(row)

    print(f"Saved CSV file to: {out_path}")


if __name__ == "__main__":
    main()
