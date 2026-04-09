import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_ground_truth(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_polygon_on_image(
    image_path: Path,
    corners: dict,
    output_path: Path,
    alpha: float = 0.15,
    color: tuple = (0, 0, 255),
):
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    points = np.array(
        [corners["tl"], corners["tr"], corners["br"], corners["bl"]], dtype=np.int32
    )

    overlay = img.copy()

    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

    return True


def process_directory(
    img_dir: Path,
    gt_data: dict,
    output_dir: Path,
    alpha: float = 0.4,
    color: tuple = (0, 0, 255),
):
    images = gt_data.get("images", [])
    processed = 0
    skipped = 0

    for item in images:
        file_name = item.get("file_name", "")
        annotations = item.get("annotations", [])

        if not annotations:
            skipped += 1
            continue

        qr_annotation = None
        for ann in annotations:
            if ann.get("category") == "qr" and "corners" in ann:
                qr_annotation = ann
                break

        if qr_annotation is None:
            skipped += 1
            continue

        img_path = img_dir / file_name
        if not img_path.exists():
            skipped += 1
            continue

        output_path = output_dir / f"gt_{file_name}"

        if draw_polygon_on_image(
            img_path, qr_annotation["corners"], output_path, alpha=alpha, color=color
        ):
            processed += 1


def main():
    parser = argparse.ArgumentParser(description="gt vis")
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument(
        "--output_dir",
        default="./output_gt",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--color",
        nargs=3,
        type=int,
        default=[0, 0, 255],
    )

    args = parser.parse_args()

    gt_path = Path(args.gt_json)
    gt_data = load_ground_truth(gt_path)

    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    color = tuple(args.color)
    process_directory(img_dir, gt_data, output_dir, args.alpha, color)


if __name__ == "__main__":
    main()
