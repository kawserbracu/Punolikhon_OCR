import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import random


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_items(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding='utf-8'))
    return data.get('items', [])


def save_items(p: Path, items: List[Dict[str, Any]]):
    p.write_text(json.dumps({'items': items}, ensure_ascii=False, indent=2), encoding='utf-8')


def split_by_images(items: List[Dict[str, Any]], train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    uniq = list({it['img_path'] for it in items})
    random.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_set = set(uniq[:n_train])
    val_set = set(uniq[n_train:n_train + n_val])
    test_set = set(uniq[n_train + n_val:])
    def filt(ss):
        return [it for it in items if it['img_path'] in ss]
    return filt(train_set), filt(val_set), filt(test_set)


def main():
    ap = argparse.ArgumentParser(description='Merge multiple converted datasets (detection/recognition) and resplit')
    ap.add_argument('--inputs', type=str, nargs='+', required=True, help='List of converted output dirs (each contains detection_*.json and recognition_*.json)')
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Merge detection
    det_all: List[Dict[str, Any]] = []
    rec_all: List[Dict[str, Any]] = []
    for d in args.inputs:
        base = Path(d)
        for name in ['detection_train.json', 'detection_val.json', 'detection_test.json']:
            p = base / name
            if p.exists():
                det_all.extend(load_items(p))
        for name in ['recognition_train.json', 'recognition_val.json', 'recognition_test.json']:
            p = base / name
            if p.exists():
                rec_all.extend(load_items(p))

    # Save merged all
    save_items(out_dir / 'detection_all.json', det_all)
    save_items(out_dir / 'recognition_all.json', rec_all)

    # Resplit detection by image
    det_train, det_val, det_test = split_by_images(det_all, train_ratio=0.7, val_ratio=0.15, seed=42)
    save_items(out_dir / 'detection_train.json', det_train)
    save_items(out_dir / 'detection_val.json', det_val)
    save_items(out_dir / 'detection_test.json', det_test)

    # Resplit recognition by parent image stem
    from pathlib import Path as P
    img_split = {}
    for it in det_train:
        img_split[P(it['img_path']).stem] = 'train'
    for it in det_val:
        img_split[P(it['img_path']).stem] = 'val'
    for it in det_test:
        img_split[P(it['img_path']).stem] = 'test'

    rec_train, rec_val, rec_test = [], [], []
    for r in rec_all:
        stem = P(r['crop_path']).stem.rsplit('_word', 1)[0]
        split = img_split.get(stem)
        if split == 'train':
            rec_train.append(r)
        elif split == 'val':
            rec_val.append(r)
        elif split == 'test':
            rec_test.append(r)

    save_items(out_dir / 'recognition_train.json', rec_train)
    save_items(out_dir / 'recognition_val.json', rec_val)
    save_items(out_dir / 'recognition_test.json', rec_test)

    # Also CSVs
    import csv
    def dump_csv(name, data):
        with (out_dir / name).open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['crop_path', 'word_text', 'char_indices', 'word_length', 'language'])
            for r in data:
                w.writerow([r['crop_path'], r.get('word_text',''), ' '.join(map(str, r.get('char_indices',[]))), r.get('word_length',0), r.get('language','')])
    dump_csv('recognition_train.csv', rec_train)
    dump_csv('recognition_val.csv', rec_val)
    dump_csv('recognition_test.csv', rec_test)

    print('Merged datasets saved to', out_dir)


if __name__ == '__main__':
    main()
