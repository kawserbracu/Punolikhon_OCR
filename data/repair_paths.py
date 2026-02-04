import argparse
from pathlib import Path
import json
from typing import Any, Dict, List
import cv2

EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_items(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding='utf-8'))
    return data.get('items', [])


def save_items(p: Path, items: List[Dict[str, Any]]):
    p.write_text(json.dumps({'items': items}, ensure_ascii=False, indent=2), encoding='utf-8')


def find_image(project_root: Path, name: str) -> Path | None:
    # exact filename search first
    for p in project_root.rglob(name):
        if p.is_file():
            return p
    # try suffix-insensitive: match by stem
    stem = Path(name).stem.lower()
    for p in project_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in EXTS and p.stem.lower() == stem:
            return p
    return None


def repair_file(json_path: Path, project_root: Path) -> Dict[str, int]:
    items = load_items(json_path)
    fixed = 0
    missing = 0
    for it in items:
        ipath = it.get('img_path') or it.get('image_path') or ''
        ok = False
        if ipath:
            img = cv2.imread(ipath)
            if img is not None:
                ok = True
        if not ok:
            # try project root / relative
            rel = Path(ipath)
            if not rel.is_absolute() and len(str(rel)) > 0:
                cand = project_root / rel
                img = cv2.imread(str(cand))
                if img is not None:
                    it['img_path'] = str(cand.resolve())
                    fixed += 1
                    continue
            # try by filename anywhere under project root
            name = Path(ipath).name
            cand2 = find_image(project_root, name)
            if cand2 is not None:
                it['img_path'] = str(cand2.resolve())
                fixed += 1
            else:
                missing += 1
    save_items(json_path, items)
    return {'fixed': fixed, 'missing': missing, 'total': len(items)}


def main():
    ap = argparse.ArgumentParser(description='Repair img_path entries in detection JSONs by resolving to actual files')
    ap.add_argument('--data_dir', type=str, required=True, help='Directory containing detection_train/val/test.json')
    ap.add_argument('--project_root', type=str, required=True, help='Project root to search for images (e.g., BAOCR)')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    project_root = Path(args.project_root)

    report = {}
    for name in ['detection_train.json', 'detection_val.json', 'detection_test.json']:
        p = data_dir / name
        if p.exists():
            stats = repair_file(p, project_root)
            report[name] = stats

    out = data_dir / 'repair_report.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Repair complete. Report at', out)


if __name__ == '__main__':
    main()
