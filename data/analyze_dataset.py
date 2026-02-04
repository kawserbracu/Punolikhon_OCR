import argparse
import json
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable

import cv2
from tqdm import tqdm


# ---------- Helpers ----------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_bengali_char(ch: str) -> bool:
    # Bengali block U+0980–U+09FF
    return 0x0980 <= ord(ch) <= 0x09FF


def is_english_char(ch: str) -> bool:
    return ('a' <= ch <= 'z') or ('A' <= ch <= 'Z')


def classify_word(word: str) -> str:
    has_bn = any(is_bengali_char(c) for c in word)
    has_en = any(is_english_char(c) for c in word)
    if has_bn and has_en:
        return 'mixed'
    if has_bn:
        return 'bengali'
    if has_en:
        return 'english'
    return 'other'


# ---------- Loaders for two formats ----------

def load_doctr_manifest(p: Path) -> Dict[str, Any]:
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict) or 'items' not in data:
        raise ValueError('Not a DocTR-style detection manifest (expected dict with key "items")')
    return data

def load_labelstudio(p: Path) -> List[Dict[str, Any]]:
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Not a Label Studio export (expected list at top-level)')
    return data


def ls_extract_image_path(task: Dict[str, Any]) -> str:
    # Classic LS
    data = task.get('data', {})
    for k in ('image', 'img', 'fp', 'path'):
        if k in data:
            return str(data[k])
    # Compact schema top-level
    for k in ('image', 'image_path', 'img', 'path'):
        if k in task:
            return str(task[k])
    return ''


# ---------- Parsing unified samples ----------

def iterate_samples_from_doctr(manifest: Dict[str, Any]) -> Iterable[Tuple[str, List[List[float]], List[str]]]:
    items = manifest.get('items', [])
    for it in items:
        img_path = it.get('image_path', '')
        boxes = it.get('boxes', [])
        texts = it.get('transcriptions', [])
        yield img_path, boxes, texts


def iterate_samples_from_labelstudio(ls_data: List[Dict[str, Any]]):
    for task in ls_data:
        img_path = ls_extract_image_path(task)

        anns = task.get('annotations', [])
        if not anns:
            # Compact schema?
            if isinstance(task.get('bbox'), list):
                # Text list may exist under various keys
                text_list = task.get('texts') or task.get('labels') or task.get('transcriptions') or []
                # Convert percent bbox to normalized [x1,y1,x2,y2] without needing image size (approximate by 100)
                boxes_norm = []
                for b in task.get('bbox'):
                    try:
                        x, y, w, h = float(b.get('x', 0.0)), float(b.get('y', 0.0)), float(b.get('width', 0.0)), float(b.get('height', 0.0))
                        boxes_norm.append([x/100.0, y/100.0, (x+w)/100.0, (y+h)/100.0])
                    except Exception:
                        boxes_norm.append([0,0,0,0])
                yield img_path, boxes_norm, list(text_list)
            else:
                yield img_path, [], []
            continue
        res = anns[0].get('result', [])
        # Pair by id where possible
        bboxes = [r for r in res if r.get('type') == 'rectanglelabels']
        texts = [r for r in res if r.get('type') == 'textarea']

        id2text = {r.get('id'): r for r in texts if r.get('id')}
        paired_texts: List[str] = []
        for br in bboxes:
            tid = br.get('id')
            txt = ''
            if tid in id2text:
                arr = id2text[tid].get('value', {}).get('text') or []
                txt = arr[0] if arr else ''
            paired_texts.append(txt)

        # For analysis we only need counts/texts; LS coords are percentages.
        # Convert percent bbox to normalized [x1,y1,x2,y2] without image access
        boxes_norm = []
        for br in bboxes:
            v = br.get('value', {})
            try:
                x, y, w, h = float(v.get('x', 0.0)), float(v.get('y', 0.0)), float(v.get('width', 0.0)), float(v.get('height', 0.0))
                boxes_norm.append([x/100.0, y/100.0, (x+w)/100.0, (y+h)/100.0])
            except Exception:
                boxes_norm.append([0,0,0,0])
        yield img_path, boxes_norm, paired_texts


# ---------- Analysis ----------

def analyze(json_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)

    # Try to load as DocTR manifest; if fails, try LS
    samples_iter = None
    total_images_set = set()

    try:
        manifest = load_doctr_manifest(json_path)
        samples_iter = iterate_samples_from_doctr(manifest)
    except Exception:
        ls_data = load_labelstudio(json_path)
        samples_iter = iterate_samples_from_labelstudio(ls_data)

    total_ann = 0
    bengali_count = 0
    english_count = 0
    mixed_count = 0
    other_count = 0

    char_freq = Counter()
    word_len_hist = Counter()

    missing_ann_images = []
    empty_label_count = 0
    corrupt_images = []

    for img_path, boxes, texts in tqdm(list(samples_iter), desc='Analyzing'):
        total_images_set.add(str(img_path))

        # Validate image if possible
        ok = False
        if isinstance(img_path, str) and len(img_path) > 0:
            # Try direct path
            img = cv2.imread(img_path)
            if img is None:
                # Try basename in same directory as JSON
                candidate = json_path.parent / Path(str(img_path)).name
                img = cv2.imread(str(candidate))
            if img is None:
                # Try common local split folders next to the JSON dir
                base = json_path.parent
                for split in ("d1", "d2", "d3", "d4"):
                    candidate2 = base / split / Path(str(img_path)).name
                    img = cv2.imread(str(candidate2))
                    if img is not None:
                        break
            if img is None:
                corrupt_images.append(str(img_path))
            else:
                ok = True

        # Count annotations
        n = len(texts) if texts is not None else 0
        total_ann += n
        if n == 0:
            missing_ann_images.append(str(img_path))

        # Text stats
        for t in texts or []:
            if t is None:
                t = ''
            if len(t.strip()) == 0:
                empty_label_count += 1
            # Word classification
            cls = classify_word(t)
            if cls == 'bengali':
                bengali_count += 1
            elif cls == 'english':
                english_count += 1
            elif cls == 'mixed':
                mixed_count += 1
            else:
                other_count += 1

            # Char frequency & length
            for ch in t:
                char_freq[ch] += 1
            word_len_hist[len(t)] += 1

    # Save character frequency CSV
    with (out_dir / 'character_frequency.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['character', 'count'])
        for ch, cnt in sorted(char_freq.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([ch, cnt])

    # Save word length distribution CSV
    with (out_dir / 'word_length_distribution.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['length', 'count'])
        for L, cnt in sorted(word_len_hist.items()):
            w.writerow([L, cnt])

    # Save summary TXT
    summary_lines = []
    summary_lines.append(f"Total images: {len(total_images_set)}")
    summary_lines.append(f"Total annotations (words): {total_ann}")
    summary_lines.append(f"Bengali words: {bengali_count}")
    summary_lines.append(f"English words: {english_count}")
    summary_lines.append(f"Mixed words: {mixed_count}")
    summary_lines.append(f"Other words: {other_count}")
    summary_lines.append("")
    summary_lines.append(f"Images with no annotations: {len(missing_ann_images)}")
    summary_lines.append(f"Empty labels: {empty_label_count}")
    summary_lines.append(f"Corrupt images (failed to read): {len(corrupt_images)}")

    with (out_dir / 'analysis_summary.txt').open('w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    # Save detail lists for debugging
    if missing_ann_images:
        with (out_dir / 'images_missing_annotations.txt').open('w', encoding='utf-8') as f:
            f.write('\n'.join(missing_ann_images))
    if corrupt_images:
        with (out_dir / 'corrupt_images.txt').open('w', encoding='utf-8') as f:
            f.write('\n'.join(corrupt_images))

    print('Analysis complete. Outputs saved to', out_dir)


def main():
    ap = argparse.ArgumentParser(description='Analyze OCR dataset JSON (DocTR manifest or Label Studio export)')
    ap.add_argument('--json_path', type=str, required=True, help='Path to detection_manifest.json or Label Studio JSON')
    ap.add_argument('--output_dir', type=str, required=True, help='Directory to save analysis outputs')
    args = ap.parse_args()

    analyze(Path(args.json_path), Path(args.output_dir))


if __name__ == '__main__':
    main()
