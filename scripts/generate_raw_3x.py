import argparse
import csv
import json
import unicodedata
from pathlib import Path

import cv2
from tqdm import tqdm


def detect_language(text: str) -> str:
    if not text:
        return "bengali"
    ascii_chars = sum(1 for c in text if ord(c) < 128 and c.isalnum())
    return "english" if ascii_chars / max(len(text), 1) > 0.5 else "bengali"


def make_rel(p: Path, relative_to: Path | None) -> str:
    if relative_to is None:
        return p.as_posix()
    try:
        return p.relative_to(relative_to).as_posix()
    except Exception:
        return p.as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Detection manifest JSON (expects {items:[...]})")
    parser.add_argument("--preproc_base", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--relative_to",
        type=str,
        default="",
        help="If set, crop_path entries are written relative to this directory (recommended: dataset root)",
    )
    args = parser.parse_args()

    preproc_base = Path(args.preproc_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rel_to = Path(args.relative_to).resolve() if args.relative_to else None

    with open(args.manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])

    raw_items = [
        it
        for it in items
        if "/raw/" in it.get("img_path", "") or "\\raw\\" in it.get("img_path", "") or it.get("source") == "raw"
    ]

    modes = ["original", "clahe", "org2"]
    all_crops = []

    for item in tqdm(raw_items, desc="Processing raw"):
        img_path_orig = item.get("img_path", "")
        img_name = Path(img_path_orig).name
        boxes = item.get("boxes", [])
        labels = item.get("labels", [])

        source = "raw"

        for mode in modes:
            img_file = preproc_base / mode / "images" / source / img_name

            if not img_file.exists():
                candidates = list((preproc_base / mode / "images" / source).glob(img_name))
                if candidates:
                    img_file = candidates[0]

            if not img_file.exists():
                continue

            img = cv2.imread(str(img_file))
            if img is None:
                continue

            h, w = img.shape[:2]

            for i, box in enumerate(boxes):
                if i >= len(labels):
                    break
                text = labels[i]

                b = [int(v) for v in box]
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_name = f"{img_file.stem}_w{i}.jpg"
                crop_out = output_dir / mode / source / "crops" / crop_name
                crop_out.parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(crop_out), crop)

                all_crops.append(
                    {
                        "crop_path": make_rel(crop_out, rel_to),
                        "word_text": unicodedata.normalize("NFC", text),
                        "language": detect_language(text),
                        "source": source,
                        "mode": mode,
                    }
                )

    out_json = output_dir / "recognition_raw.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_crops, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "recognition_raw.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "word_text", "language", "source", "mode"])
        writer.writeheader()
        writer.writerows(all_crops)

    print(f"Generated {len(all_crops)} raw crops")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
