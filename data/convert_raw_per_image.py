import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_box(points: List[List[float]]) -> Tuple[int, int, int, int]:
    # Accept 2 points (x1,y1),(x2,y2) or arbitrary list -> min/max
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def abs_to_norm(x1: int, y1: int, x2: int, y2: int, iw: int, ih: int) -> List[float]:
    x1 = max(0, min(iw - 1, x1)); x2 = max(0, min(iw - 1, x2))
    y1 = max(0, min(ih - 1, y1)); y2 = max(0, min(ih - 1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1 / iw, y1 / ih, x2 / iw, y2 / ih]


def main():
    ap = argparse.ArgumentParser(description='Convert per-image JSON annotations (BanglaWriting-like) to DocTR manifests')
    ap.add_argument('--raw_dir', type=str, required=True, help='Directory containing images and per-image JSONs')
    ap.add_argument('--output_dir', type=str, required=True, help='Directory to write detection_all.json and recognition_all.json')
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    det_items: List[Dict[str, Any]] = []
    rec_items: List[Dict[str, Any]] = []

    crops_dir = out_dir / 'crops'
    ensure_dir(crops_dir)

    json_files = sorted(raw_dir.glob('*.json'))
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
        except Exception:
            continue
        # image path + size
        iw = int(float(data.get('imageWidth', 0)))
        ih = int(float(data.get('imageHeight', 0)))
        # Prefer sibling image with same stem if path missing
        img_path = raw_dir / (data.get('imagePath') or (jf.stem + '.jpg'))
        if not img_path.exists():
            # try common extensions
            for ext in ('.jpeg', '.png', '.bmp', '.JPG', '.PNG'):
                alt = raw_dir / (jf.stem + ext)
                if alt.exists():
                    img_path = alt
                    break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if iw <= 0 or ih <= 0:
            ih, iw = img.shape[:2]
        shapes = data.get('shapes', []) or []
        boxes_norm: List[List[float]] = []
        labels: List[str] = []
        for sh in shapes:
            lbl = sh.get('label') or ''
            pts = sh.get('points') or []
            if not pts:
                continue
            x1, y1, x2, y2 = to_box(pts)
            # Skip empty/invalid boxes
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue
            bn = abs_to_norm(x1, y1, x2, y2, iw, ih)
            boxes_norm.append(bn)
            labels.append(lbl)
        if not boxes_norm:
            continue
        det_items.append({
            'img_path': str(img_path.resolve()),
            'img_dimensions': (ih, iw),
            'boxes': boxes_norm,
            'labels': labels,
        })
        # Write crops + rec items
        for idx, (bn, text) in enumerate(zip(boxes_norm, labels)):
            xmin = int(round(bn[0] * iw)); ymin = int(round(bn[1] * ih))
            xmax = int(round(bn[2] * iw)); ymax = int(round(bn[3] * ih))
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            crop_name = f"{img_path.stem}_word{idx+1}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            rec_items.append({
                'crop_path': str(crop_path.resolve()),
                'word_text': text or '',
                'language': 'bengali' if any(0x0980 <= ord(c) <= 0x09FF for c in (text or '')) else 'english',
            })

    # Save outputs
    (out_dir / 'detection_all.json').write_text(json.dumps({'items': det_items}, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'recognition_all.json').write_text(json.dumps({'items': rec_items}, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Converted per-image RAW annotations:')
    print('  detection_all items:', len(det_items))
    print('  recognition_all items:', len(rec_items))
    print('  output_dir:', out_dir)


if __name__ == '__main__':
    main()
