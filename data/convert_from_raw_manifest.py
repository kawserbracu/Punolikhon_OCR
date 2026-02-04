import argparse
import csv
import sys
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_detection_manifest(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def load_recognition_csv(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Increase field size limit to handle long fields
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(int(1e9))
    with p.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def split_by_images(items: List[Dict[str, Any]], train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    uniq = list({it['image_path'] for it in items})
    random.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_set = set(uniq[:n_train])
    val_set = set(uniq[n_train:n_train + n_val])
    test_set = set(uniq[n_train + n_val:])
    def filt(ss):
        return [it for it in items if it['image_path'] in ss]
    return filt(train_set), filt(val_set), filt(test_set)


def main():
    ap = argparse.ArgumentParser(description='Convert aggregated RAW manifests into detection/recognition splits')
    ap.add_argument('--raw_dir', type=str, required=True, help='Directory that contains detection_manifest.json and recognition_manifest.csv')
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    det_path = raw_dir / 'detection_manifest.json'
    rec_path = raw_dir / 'recognition_manifest.csv'
    if not det_path.exists():
        raise FileNotFoundError(str(det_path))
    if not rec_path.exists():
        raise FileNotFoundError(str(rec_path))

    det = load_detection_manifest(det_path)
    det_items = det.get('items', [])

    # Save unified detection_all.json in our expected fields
    det_all: List[Dict[str, Any]] = []
    for it in det_items:
        # normalize fields to match convert_dataset output
        det_all.append({
            'img_path': it.get('image_path', ''),
            'img_dimensions': (it.get('height', 0), it.get('width', 0)),
            'boxes': it.get('boxes', []),
            'labels': it.get('transcriptions', []),
        })
    (out_dir / 'detection_all.json').write_text(json.dumps({'items': det_all}, ensure_ascii=False, indent=2), encoding='utf-8')

    # Recognition
    rec_rows = load_recognition_csv(rec_path)
    rec_all: List[Dict[str, Any]] = []
    for r in rec_rows:
        text = r.get('text') or r.get('word_text') or ''
        rec_all.append({
            'crop_path': r.get('path') or r.get('crop_path') or '',
            'word_text': text,
            'char_indices': [],
            'word_length': len(text),
            'language': 'bengali' if any(0x0980 <= ord(c) <= 0x09FF for c in text) else 'english',
        })
    (out_dir / 'recognition_all.json').write_text(json.dumps({'items': rec_all}, ensure_ascii=False, indent=2), encoding='utf-8')

    # Split detection by images; map recognition by image stem
    det_train, det_val, det_test = split_by_images([{**it, 'image_path': it['img_path']} for it in det_all])

    def save_items(name, items):
        (out_dir / name).write_text(json.dumps({'items': items}, ensure_ascii=False, indent=2), encoding='utf-8')

    save_items('detection_train.json', det_train)
    save_items('detection_val.json', det_val)
    save_items('detection_test.json', det_test)

    # recognition splits via image stem
    img_split = {}
    from pathlib import Path as P
    for it in det_train:
        img_split[P(it['img_path']).stem] = 'train'
    for it in det_val:
        img_split[P(it['img_path']).stem] = 'val'
    for it in det_test:
        img_split[P(it['img_path']).stem] = 'test'

    rec_train, rec_val, rec_test = [], [], []
    for r in rec_all:
        stem = P(r['crop_path']).stem.rsplit('_', 1)[0]
        # try to match most common stem pattern; if not found, put into train by default
        split = None
        for k, v in img_split.items():
            if k in stem:
                split = v
                break
        if split == 'val':
            rec_val.append(r)
        elif split == 'test':
            rec_test.append(r)
        else:
            rec_train.append(r)

    save_items('recognition_train.json', rec_train)
    save_items('recognition_val.json', rec_val)
    save_items('recognition_test.json', rec_test)

    # CSVs
    import csv as _csv
    def dump_csv(name, data):
        with (out_dir / name).open('w', encoding='utf-8', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['crop_path', 'word_text', 'char_indices', 'word_length', 'language'])
            for r in data:
                w.writerow([r['crop_path'], r['word_text'], ' '.join(map(str, r.get('char_indices', []))), r['word_length'], r['language']])
    dump_csv('recognition_train.csv', rec_train)
    dump_csv('recognition_val.csv', rec_val)
    dump_csv('recognition_test.csv', rec_test)

    print('Converted RAW manifests into splits at', out_dir)


if __name__ == '__main__':
    main()
