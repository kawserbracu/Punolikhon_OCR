import argparse
import csv
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import cv2
from tqdm import tqdm


def load_labelstudio_annotations(json_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("items", [])


def extract_boxes_and_texts(task: Dict[str, Any], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if "bbox" in task and "transcription" in task:
        bbox_list = task.get("bbox", [])
        text_list = task.get("transcription", [])

        if bbox_list and isinstance(bbox_list, list) and len(bbox_list) > 0 and isinstance(bbox_list[0], dict):
            for i, box_dict in enumerate(bbox_list):
                x_pct = box_dict.get("x", 0)
                y_pct = box_dict.get("y", 0)
                w_pct = box_dict.get("width", 0)
                h_pct = box_dict.get("height", 0)

                x1 = int(round(x_pct * img_w / 100))
                y1 = int(round(y_pct * img_h / 100))
                x2 = int(round((x_pct + w_pct) * img_w / 100))
                y2 = int(round((y_pct + h_pct) * img_h / 100))

                text = text_list[i] if i < len(text_list) else ""
                results.append({"box": [x1, y1, x2, y2], "text": text})
            return results

        if bbox_list and isinstance(bbox_list, list) and len(bbox_list) == 4 and isinstance(bbox_list[0], (int, float)):
            x1, y1, x2, y2 = bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3]
            if x2 < x1 or y2 < y1:
                x2 = x1 + bbox_list[2]
                y2 = y1 + bbox_list[3]
            results.append({"box": [int(x1), int(y1), int(x2), int(y2)], "text": str(text_list) if not isinstance(text_list, str) else text_list})
            return results

    annotations = task.get("annotations", [])

    text_map: Dict[str, str] = {}
    for ann in annotations:
        for r in ann.get("result", []):
            if r.get("type") == "textarea":
                rid = r.get("id", "")
                text_val = r.get("value", {}).get("text", [""])[0]
                text_map[rid] = text_val

    for ann in annotations:
        for r in ann.get("result", []):
            if r.get("type") == "rectanglelabels":
                val = r.get("value", {})
                x_pct = val.get("x", 0)
                y_pct = val.get("y", 0)
                w_pct = val.get("width", 0)
                h_pct = val.get("height", 0)

                x1 = int(round(x_pct * img_w / 100))
                y1 = int(round(y_pct * img_h / 100))
                x2 = int(round((x_pct + w_pct) * img_w / 100))
                y2 = int(round((y_pct + h_pct) * img_h / 100))

                rid = r.get("id", "")
                text = text_map.get(rid, "")
                results.append({"box": [x1, y1, x2, y2], "text": text})

    return results


def crop_and_save(img: Any, box: List[int], out_path: Path) -> bool:
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return True


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


def process_source(
    source_name: str,
    annotations_path: Path,
    preproc_base: Path,
    output_base: Path,
    modes: List[str],
    relative_to: Path | None,
) -> List[Dict[str, Any]]:
    tasks = load_labelstudio_annotations(annotations_path)

    all_crops: List[Dict[str, Any]] = []

    for mode in modes:
        images_dir = preproc_base / mode / "images" / source_name
        crops_dir = output_base / mode / source_name / "crops"

        if not images_dir.exists():
            continue

        crops_dir.mkdir(parents=True, exist_ok=True)

        for task in tqdm(tasks, desc=f"{source_name}:{mode}", leave=False):
            img_hint = task.get("image", "")
            if not img_hint:
                data = task.get("data", {})
                img_hint = data.get("image", "") or data.get("img", "")
            if not img_hint:
                continue

            img_name = Path(img_hint).name
            img_path = images_dir / img_name

            if not img_path.exists():
                for ext in [".jpg", ".jpeg", ".png"]:
                    alt = images_dir / (Path(img_name).stem + ext)
                    if alt.exists():
                        img_path = alt
                        break

            if not img_path.exists():
                import re

                uuid_pattern = r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"
                matches = re.findall(uuid_pattern, img_name.lower())
                if matches:
                    for uuid_match in matches:
                        for ext in [".jpg", ".jpeg", ".png"]:
                            candidate = images_dir / f"{uuid_match}{ext}"
                            if candidate.exists():
                                img_path = candidate
                                break
                        if img_path.exists():
                            break

            if not img_path.exists():
                hint_stem = Path(img_name).stem.lower()
                for disk_file in images_dir.glob("*"):
                    if disk_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                        continue
                    disk_stem = disk_file.stem.lower()
                    if disk_stem in hint_stem and len(disk_stem) > 3 and hint_stem.endswith(disk_stem):
                        img_path = disk_file
                        break

            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            boxes_texts = extract_boxes_and_texts(task, w, h)

            for idx, bt in enumerate(boxes_texts):
                crop_name = f"{img_path.stem}_word{idx + 1}.jpg"
                crop_path = crops_dir / crop_name

                if crop_and_save(img, bt["box"], crop_path):
                    text = bt["text"]
                    all_crops.append(
                        {
                            "crop_path": make_rel(crop_path, relative_to),
                            "word_text": unicodedata.normalize("NFC", text) if text else "",
                            "language": detect_language(text),
                            "source": source_name,
                            "mode": mode,
                        }
                    )

    return all_crops


def main():
    ap = argparse.ArgumentParser(description="Generate 3x GT crops from preprocessed images")
    ap.add_argument("--preproc_base", type=str, required=True, help="Base directory containing original/clahe/org2 folders")
    ap.add_argument("--annotations_dir", type=str, required=True, help="Directory containing annotation JSON files")
    ap.add_argument("--output_base", type=str, required=True, help="Output directory for crops")
    ap.add_argument("--modes", type=str, default="original,clahe,org2")
    ap.add_argument(
        "--relative_to",
        type=str,
        default="",
        help="If set, crop_path entries are written relative to this directory (recommended: dataset root)",
    )
    args = ap.parse_args()

    preproc_base = Path(args.preproc_base)
    annotations_dir = Path(args.annotations_dir)
    output_base = Path(args.output_base)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    rel_to = Path(args.relative_to).resolve() if args.relative_to else None

    output_base.mkdir(parents=True, exist_ok=True)

    source_annotations = {
        "d1": annotations_dir / "annotations" / "d1_annotations.json",
        "d2": annotations_dir / "annotations" / "d2_annotations.json",
        "d3": annotations_dir / "annotations" / "d3_annotations.json",
        "d4": annotations_dir / "annotations" / "d4_annotations.json",
        "d5": annotations_dir / "d5_annotations_ls.json",
        "d6": annotations_dir / "d6_annotations_ls.json",
    }

    all_crops: List[Dict[str, Any]] = []

    for source, ann_path in source_annotations.items():
        if ann_path.exists():
            all_crops.extend(process_source(source, ann_path, preproc_base, output_base, modes, rel_to))

    manifest_path = output_base / "recognition_all.json"
    manifest_path.write_text(json.dumps(all_crops, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_base / "recognition_all.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "word_text", "language", "source", "mode"])
        writer.writeheader()
        writer.writerows(all_crops)

    print(f"Generated {len(all_crops)} crops")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
