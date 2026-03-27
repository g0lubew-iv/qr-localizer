import argparse
import json
import re
from pathlib import Path


def natural_key(p: Path):
    parts = re.split(r"(\d+)", p.stem)
    key = []
    for x in parts:
        if x.isdigit():
            key.append(int(x))
        else:
            key.append(x.lower())
    return key


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_qr_polygon(labelme: dict, label: str = "qr"):
    shapes = labelme.get("shapes", [])
    for s in shapes:
        if s.get("shape_type") == "polygon" and s.get("label") == label:
            return s
    for s in shapes:
        if s.get("shape_type") == "polygon":
            return s
    return None


def corners_from_click_order(pts):
    if len(pts) != 4:
        raise ValueError(f"Expected 4 points, got {len(pts)}")
    tl, tr, br, bl = pts
    return {
        "tl": [float(tl[0]), float(tl[1])],
        "tr": [float(tr[0]), float(tr[1])],
        "br": [float(br[0]), float(br[1])],
        "bl": [float(bl[0]), float(bl[1])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--category", default="qr")
    ap.add_argument("--label", default="qr")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    ann_dir = Path(args.ann_dir)
    out_path = Path(args.out)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in exts], key=natural_key
    )

    images_out = []
    img_id = 1
    ann_id = 1

    for img_path in img_files:
        ann_path = ann_dir / f"{img_path.stem}.json"
        if not ann_path.exists():
            msg = f"Missing annotation for {img_path.name}: expected {ann_path}"
            if args.strict:
                raise SystemExit(msg)
            print("Warning:", msg, "- skipping")
            continue

        lm = load_json(ann_path)
        w = lm.get("imageWidth")
        h = lm.get("imageHeight")
        file_name = lm.get("imagePath", img_path.name)

        shape = find_qr_polygon(lm, label=args.label)
        if shape is None:
            msg = f"No polygon found in {ann_path.name}"
            if args.strict:
                raise SystemExit(msg)
            print("Warning:", msg, "- skipping")
            continue

        pts = shape.get("points", [])
        corners = corners_from_click_order(pts)

        images_out.append(
            {
                "id": img_id,
                "file_name": file_name,
                "width": int(w) if w is not None else None,
                "height": int(h) if h is not None else None,
                "annotations": [
                    {"id": ann_id, "category": args.category, "corners": corners}
                ],
            }
        )

        img_id += 1
        ann_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"images": images_out}, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(images_out)} images to {out_path}")


if __name__ == "__main__":
    main()
