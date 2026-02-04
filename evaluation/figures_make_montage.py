import argparse
from pathlib import Path
import cv2
import json
from typing import List
import numpy as np


def grid(images: List[np.ndarray], cols: int = 2, pad: int = 6, bg=(240, 240, 240)) -> np.ndarray:
    if not images:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    norm = [cv2.copyMakeBorder(cv2.resize(im, (w, h)), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg) for im in images]
    rows = (len(images) + cols - 1) // cols
    row_imgs = []
    for r in range(rows):
        items = norm[r * cols:(r + 1) * cols]
        if len(items) < cols:
            for _ in range(cols - len(items)):
                items.append(np.full_like(norm[0], 255))
        row_imgs.append(np.concatenate(items, axis=1))
    return np.concatenate(row_imgs, axis=0)


def overlay(results_json: Path, out_dir: Path) -> List[Path]:
    data = json.loads(results_json.read_text(encoding='utf-8'))
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for img_path, items in data.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        vis = img.copy()
        for it in items:
            b = it.get('box', None)
            txt = it.get('text', '')
            if not b:
                continue
            x1 = int(round(b[0] * W)); y1 = int(round(b[1] * H))
            x2 = int(round(b[2] * W)); y2 = int(round(b[3] * H))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            if txt:
                cv2.putText(vis, txt, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1, cv2.LINE_AA)
        out_p = out_dir / (Path(img_path).stem + '_overlay.jpg')
        cv2.imwrite(str(out_p), vis)
        outs.append(out_p)
    return outs


def main():
    ap = argparse.ArgumentParser(description='Create side-by-side montages from two inference results folders (overlays)')
    ap.add_argument('--left_results', type=str, required=True, help='Folder A with results.json and *_vis images or source images')
    ap.add_argument('--right_results', type=str, required=True, help='Folder B with results.json and *_vis images or source images')
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--limit', type=int, default=20)
    args = ap.parse_args()

    left = Path(args.left_results)
    right = Path(args.right_results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build overlays if results.json exists
    left_json = left / 'results.json'
    right_json = right / 'results.json'
    left_vis_dir = left / 'overlays'
    right_vis_dir = right / 'overlays'
    l_imgs = []
    r_imgs = []

    if left_json.exists():
        left_paths = overlay(left_json, left_vis_dir)
        l_imgs = left_paths
    else:
        l_imgs = sorted([p for p in left.rglob('*.jpg')])

    if right_json.exists():
        right_paths = overlay(right_json, right_vis_dir)
        r_imgs = right_paths
    else:
        r_imgs = sorted([p for p in right.rglob('*.jpg')])

    # Pair by stem
    common = {}
    for p in l_imgs:
        common[Path(p).stem] = {'l': p}
    for p in r_imgs:
        st = Path(p).stem
        common.setdefault(st, {})['r'] = p

    pairs = [(k, v['l'], v.get('r')) for k, v in common.items() if 'l' in v and 'r' in v]
    pairs = pairs[: args.limit]

    for stem, lp, rp in pairs:
        limg = cv2.imread(str(lp))
        rimg = cv2.imread(str(rp))
        if limg is None or rimg is None:
            continue
        # Uniform height
        H = 700
        def resize_h(im):
            h, w = im.shape[:2]
            nw = int(round(w * (H / h)))
            return cv2.resize(im, (nw, H))
        limg = resize_h(limg)
        rimg = resize_h(rimg)
        cat = grid([limg, rimg], cols=2)
        cv2.putText(cat, 'LEFT', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(cat, 'RIGHT', (cat.shape[1]//2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
        outp = out_dir / f'{stem}_montage.jpg'
        cv2.imwrite(str(outp), cat)

    print('Saved montages to', out_dir)


if __name__ == '__main__':
    main()
