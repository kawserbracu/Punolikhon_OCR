import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
import cv2
from tqdm import tqdm

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data.tokenizer import BengaliWordOCRTokenizer


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_labelstudio(p: Path) -> List[Dict[str, Any]]:
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Label Studio export must be a top-level list')
    return data


def extract_image_path(task: Dict[str, Any]) -> str:
    data = task.get('data', {})
    # Common LS keys under data
    for k in ('image', 'img', 'fp', 'path', 'image_url', 'imagePath', 'img_path', 'file_upload', 'original_filename'):
        if k in data and data[k]:
            return str(data[k])
    # Sometimes nested under meta
    meta = task.get('meta', {})
    for k in ('image', 'img', 'path', 'file_upload', 'original_filename'):
        if k in meta and meta[k]:
            return str(meta[k])
    # Top-level fallbacks
    for k in ('image', 'image_path', 'img', 'path', 'image_url', 'file_upload', 'original_filename'):
        if k in task and task[k]:
            return str(task[k])
    return ''


def ls_pairs(task: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    anns = task.get('annotations', [])
    if not anns:
        return [], []
    res = anns[0].get('result', [])
    bboxes = [r for r in res if r.get('type') == 'rectanglelabels']
    texts = [r for r in res if r.get('type') == 'textarea']
    id2text = {r.get('id'): r for r in texts if r.get('id')}
    paired_texts: List[str] = []
    used_bboxes: List[Dict[str, Any]] = []
    for br in bboxes:
        tid = br.get('id')
        txt = ''
        if tid in id2text:
            arr = id2text[tid].get('value', {}).get('text') or []
            txt = arr[0] if arr else ''
        paired_texts.append(txt)
        used_bboxes.append(br)
    return used_bboxes, paired_texts


def ls_compact(task: Dict[str, Any]) -> Tuple[List[Dict[str, float]], List[str]]:
    boxes = []
    texts = task.get('texts') or task.get('labels') or task.get('transcriptions') or []
    for b in task.get('bbox', []) or []:
        if not isinstance(b, dict):
            continue
        boxes.append({
            'x': float(b.get('x', 0.0)),
            'y': float(b.get('y', 0.0)),
            'width': float(b.get('width', 0.0)),
            'height': float(b.get('height', 0.0))
        })
    return boxes, list(texts)


def percent_to_abs(v: Dict[str, float], iw: int, ih: int) -> Tuple[int, int, int, int]:
    x = float(v.get('x', 0.0)) * iw / 100.0
    y = float(v.get('y', 0.0)) * ih / 100.0
    w = float(v.get('width', 0.0)) * iw / 100.0
    h = float(v.get('height', 0.0)) * ih / 100.0
    xmin = max(0, int(round(x)))
    ymin = max(0, int(round(y)))
    xmax = min(iw - 1, int(round(x + w)))
    ymax = min(ih - 1, int(round(y + h)))
    return xmin, ymin, xmax, ymax


def abs_to_norm(box: Tuple[int, int, int, int], iw: int, ih: int) -> List[float]:
    xmin, ymin, xmax, ymax = box
    return [xmin / iw, ymin / ih, xmax / iw, ymax / ih]


def resolve_local_image(images_dir: Path, img_hint: str) -> Path | None:
    # Prefer basename and search in images_dir
    name = Path(img_hint).name
    cand = images_dir / name
    if cand.exists():
        return cand
    # Try case-insensitive match (direct children)
    try:
        for p in images_dir.iterdir():
            if p.is_file() and p.name.lower() == name.lower():
                return p
    except Exception:
        pass
    # Try recursive search by exact name
    try:
        for p in images_dir.rglob(name):
            if p.is_file():
                return p
    except Exception:
        pass
    # Try recursive search by stem case-insensitive
    stem = Path(name).stem.lower()
    try:
        for p in images_dir.rglob('*'):
            if p.is_file() and p.stem.lower() == stem:
                return p
    except Exception:
        pass
    # Try same stem with alternative extensions (useful after bulk .jpg conversion)
    alt_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    try:
        for p in images_dir.rglob('*'):
            if p.is_file() and p.stem.lower() == stem and p.suffix.lower() in alt_exts:
                return p
    except Exception:
        pass
    # Try suffix match: local file might be the tail after a prefix (e.g., 'feefbf94-Unknown-1.jpeg')
    name_lower = name.lower()
    try:
        for p in images_dir.rglob('*'):
            if p.is_file() and name_lower.endswith(p.name.lower()):
                return p
    except Exception:
        pass
    # Try post-hyphen remainder
    if '-' in name:
        remainder = name.split('-', 1)[1]
        try:
            for p in images_dir.rglob(remainder):
                if p.is_file():
                    return p
        except Exception:
            pass
    # Try GUID-suffix match: many LS paths have extra leading token, local file may be the trailing GUID
    import re
    m = re.search(r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})', name)
    if m:
        guid = m.group(1).lower()
        try:
            for p in images_dir.rglob('*'):
                if p.is_file() and guid in p.name.lower():
                    return p
        except Exception:
            pass
    return None


def index_images_by_size(images_dir: Path) -> Dict[Tuple[int, int], List[Path]]:
    idx: Dict[Tuple[int, int], List[Path]] = {}
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    for p in images_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            img = cv2.imread(str(p))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            idx.setdefault((iw, ih), []).append(p)
    return idx


def get_size_from_task(task: Dict[str, Any]) -> Tuple[int, int] | None:
    # Try compact schema bboxes carrying original_width/height
    bbs = task.get('bbox') or []
    for b in bbs:
        ow = int(float(b.get('original_width', 0))) if b.get('original_width') is not None else 0
        oh = int(float(b.get('original_height', 0))) if b.get('original_height') is not None else 0
        if ow > 0 and oh > 0:
            return (ow, oh)
    return None


def convert_detection(labelstudio_json: Path, images_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
    ensure_dir(out_dir)
    ls_data = load_labelstudio(labelstudio_json)

    items: List[Dict[str, Any]] = []
    size_index = index_images_by_size(images_dir)
    skipped_no_image = 0
    skipped_read_fail = 0
    skipped_no_ann = 0
    for task in tqdm(ls_data, desc='Convert (detection)'):
        img_rel = extract_image_path(task)
        img_path = resolve_local_image(images_dir, img_rel) or resolve_local_image(images_dir, Path(img_rel).name)
        if img_path is None:
            # Try size-based resolution
            wh = get_size_from_task(task)
            if wh and wh in size_index:
                cands = size_index[wh]
                if len(cands) == 1:
                    img_path = cands[0]
                elif len(cands) > 1:
                    # deterministic pick by name
                    img_path = sorted(cands)[0]
        if img_path is None:
            skipped_no_image += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped_read_fail += 1
            continue
        ih, iw = img.shape[:2]

        boxes_norm: List[List[float]] = []
        labels: List[str] = []
        anns = task.get('annotations', [])
        if anns:
            bboxes, texts = ls_pairs(task)
            for br, txt in zip(bboxes, texts):
                v = br.get('value', {})
                box_abs = percent_to_abs(v, iw, ih)
                boxes_norm.append(abs_to_norm(box_abs, iw, ih))
                labels.append(txt)
        elif isinstance(task.get('bbox'), list):
            bboxes, texts = ls_compact(task)
            for v, txt in zip(bboxes, texts or [""] * len(bboxes)):
                box_abs = percent_to_abs(v, iw, ih)
                boxes_norm.append(abs_to_norm(box_abs, iw, ih))
                labels.append(txt)
        else:
            skipped_no_ann += 1
            continue

        items.append({
            'img_path': str(img_path.resolve()),
            'img_dimensions': (ih, iw),
            'boxes': boxes_norm,
            'labels': labels,
        })

    # Save detection_all and a small debug report
    out_json = out_dir / 'detection_all.json'
    with out_json.open('w', encoding='utf-8') as f:
        json.dump({'items': items}, f, ensure_ascii=False, indent=2)
    (out_dir / 'conversion_debug.txt').write_text(
        f"skipped_no_image={skipped_no_image}\n" \
        f"skipped_read_fail={skipped_read_fail}\n" \
        f"skipped_no_ann={skipped_no_ann}\n" \
        f"kept_items={len(items)}\n", encoding='utf-8')
    return items


def convert_recognition(items: List[Dict[str, Any]], out_dir: Path, vocab_path: Path) -> List[Dict[str, Any]]:
    ensure_dir(out_dir)
    crops_dir = out_dir / 'crops'
    ensure_dir(crops_dir)

    tok = BengaliWordOCRTokenizer()
    if vocab_path.exists():
        tok.load_vocab(vocab_path)
    else:
        tok.save_vocab(vocab_path)

    def detect_language(text: str) -> str:
        if any(0x0980 <= ord(c) <= 0x09FF for c in text):
            return 'bengali'
        return 'english'

    rec_items: List[Dict[str, Any]] = []
    for it in tqdm(items, desc='Convert (recognition)'):
        img_path = Path(it['img_path'])
        ih, iw = it['img_dimensions']
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for idx, (bn, text) in enumerate(zip(it['boxes'], it['labels'])):
            xmin = int(round(bn[0] * iw))
            ymin = int(round(bn[1] * ih))
            xmax = int(round(bn[2] * iw))
            ymax = int(round(bn[3] * ih))
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            crop_name = f"{img_path.stem}_word{idx+1}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            enc = tok.encode_word(text or "")
            rec_items.append({
                'crop_path': str(crop_path.resolve()),
                'word_text': text or "",
                'char_indices': enc,
                'word_length': len(enc),
                'language': detect_language(text or ""),
            })

    out_json = out_dir / 'recognition_all.json'
    with out_json.open('w', encoding='utf-8') as f:
        json.dump({'items': rec_items}, f, ensure_ascii=False, indent=2)

    # Also write CSV
    import csv
    csv_path = out_dir / 'recognition_all.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['crop_path', 'word_text', 'char_indices', 'word_length', 'language'])
        for r in rec_items:
            w.writerow([r['crop_path'], r['word_text'], ' '.join(map(str, r['char_indices'])), r['word_length'], r['language']])

    return rec_items


def split_by_images(items: List[Dict[str, Any]], train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    # Unique images
    uniq = list({it['img_path'] for it in items})
    random.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_set = set(uniq[:n_train])
    val_set = set(uniq[n_train:n_train + n_val])
    test_set = set(uniq[n_train + n_val:])

    def filter_items(paths_set):
        return [it for it in items if it['img_path'] in paths_set]

    return filter_items(train_set), filter_items(val_set), filter_items(test_set)


def save_split_detection(train, val, test, out_dir: Path):
    def dump(name, data):
        with (out_dir / name).open('w', encoding='utf-8') as f:
            json.dump({'items': data}, f, ensure_ascii=False, indent=2)
    dump('detection_train.json', train)
    dump('detection_val.json', val)
    dump('detection_test.json', test)


def save_split_recognition(rec_all: List[Dict[str, Any]], det_train, det_val, det_test, out_dir: Path):
    img_to_split = {}
    for it in det_train:
        img_to_split[it['img_path']] = 'train'
    for it in det_val:
        img_to_split[it['img_path']] = 'val'
    for it in det_test:
        img_to_split[it['img_path']] = 'test'

    rec_train, rec_val, rec_test = [], [], []
    for r in rec_all:
        # crop belongs to source image by stem prefix
        stem = Path(r['crop_path']).stem.rsplit('_word', 1)[0]
        # find any item whose img_path stem matches
        split = None
        for src, s in img_to_split.items():
            if Path(src).stem == stem:
                split = s
                break
        if split == 'train':
            rec_train.append(r)
        elif split == 'val':
            rec_val.append(r)
        elif split == 'test':
            rec_test.append(r)

    def dump(name, data):
        with (out_dir / name).open('w', encoding='utf-8') as f:
            json.dump({'items': data}, f, ensure_ascii=False, indent=2)
    dump('recognition_train.json', rec_train)
    dump('recognition_val.json', rec_val)
    dump('recognition_test.json', rec_test)

    # CSVs
    import csv
    def dump_csv(name, data):
        with (out_dir / name).open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['crop_path', 'word_text', 'char_indices', 'word_length', 'language'])
            for r in data:
                w.writerow([r['crop_path'], r['word_text'], ' '.join(map(str, r['char_indices'])), r['word_length'], r['language']])
    dump_csv('recognition_train.csv', rec_train)
    dump_csv('recognition_val.csv', rec_val)
    dump_csv('recognition_test.csv', rec_test)


def main():
    ap = argparse.ArgumentParser(description='Convert Label Studio to DocTR-like detection/recognition with splits')
    ap.add_argument('--labelstudio_json', type=str, required=True)
    ap.add_argument('--images_dir', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--vocab_path', type=str, default=None)
    args = ap.parse_args()

    labelstudio_json = Path(args.labelstudio_json)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Detection
    det_items = convert_detection(labelstudio_json, images_dir, out_dir)

    # Splits (by image)
    train_det, val_det, test_det = split_by_images(det_items, train_ratio=0.7, val_ratio=0.15, seed=42)
    save_split_detection(train_det, val_det, test_det, out_dir)

    # Recognition
    vocab_path = Path(args.vocab_path) if args.vocab_path else (out_dir / 'vocab.json')
    rec_all = convert_recognition(det_items, out_dir, vocab_path)
    save_split_recognition(rec_all, train_det, val_det, test_det, out_dir)

    print('Conversion complete:')
    print(f"  Detection items: {len(det_items)}")
    print(f"  Recognition crops: {len(rec_all)}")
    print(f"  Output dir: {out_dir}")


if __name__ == '__main__':
    main()
