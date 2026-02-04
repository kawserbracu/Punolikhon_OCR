import argparse
import shutil
from pathlib import Path
from typing import Tuple

import cv2
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to3(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(gray)
    return to3(out)


def apply_highboost(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    high = gray + 1.5 * (gray - blurred)
    high = high.clip(0, 255).astype('uint8')
    return to3(high)


def apply_org2(img):
    """MNIST-like cleanup: background normalization + Otsu threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 21)
    normalized = cv2.normalize(cv2.subtract(gray, bg), None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return to3(th)


def process_one(src: Path, dst_img_dir: Path, mode: str) -> Tuple[str, bool]:
    try:
        img = cv2.imread(str(src))
        if img is None:
            return str(src), False
        if mode == 'original':
            out = img
        elif mode == 'org2':
            out = apply_org2(img)
        elif mode == 'clahe':
            out = apply_clahe(img)
        elif mode == 'clahe_org2':
            out = apply_clahe(apply_org2(img))
        elif mode == 'highboost':
            out = apply_highboost(img)
        else:
            return str(src), False
        dst = dst_img_dir / f"{src.stem}.jpg"
        cv2.imwrite(str(dst), out)
        return str(dst), True
    except Exception:
        return str(src), False


def preprocess_split(input_dir: Path, output_base: Path, mode: str):
    split_name = Path(input_dir).name
    out_split = output_base / mode
    img_out = out_split / 'images' / split_name
    ensure_dir(img_out)

    images = [p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}]
    with Pool(max(1, cpu_count() - 1)) as pool:
        for _ in tqdm(pool.imap_unordered(partial(process_one, dst_img_dir=img_out, mode=mode), images), total=len(images), desc=f'{mode}:{split_name}'):
            pass


def main():
    ap = argparse.ArgumentParser(description='Apply preprocessing to images and crops')
    ap.add_argument('--input_dir', type=str, required=True, help='Directory that contains split folders like d1,d2,d3,d4 or a single images folder')
    ap.add_argument('--output_base', type=str, required=True, help='Base output dir to create original/clahe/highboost subfolders')
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_base = Path(args.output_base)

    # If input_dir contains d1..d4, process each; else treat input_dir itself as images folder
    splits = []
    for s in ('d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'raw'):
        cand = input_dir / s
        if cand.exists() and cand.is_dir():
            splits.append(cand)
    if not splits:
        splits = [input_dir]

    for mode in ('original', 'org2', 'clahe', 'clahe_org2', 'highboost'):
        for s in splits:
            preprocess_split(s, output_base, mode)


if __name__ == '__main__':
    main()
