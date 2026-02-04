import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_name_index(root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in root.rglob('*'):
        if p.is_file():
            idx[p.name] = p
    return idx


def main():
    ap = argparse.ArgumentParser(description='Rebase detection manifest img_path entries to a new image root (e.g., CLAHE images)')
    ap.add_argument('--input_json', type=str, required=True, help='Path to detection_{train,val,test}.json')
    ap.add_argument('--new_root', type=str, required=True, help='Root directory containing alternative images (searched recursively)')
    ap.add_argument('--output_json', type=str, required=True)
    args = ap.parse_args()

    in_path = Path(args.input_json)
    out_path = Path(args.output_json)
    new_root = Path(args.new_root)

    data = json.loads(in_path.read_text(encoding='utf-8'))
    items: List[Dict[str, Any]] = data.get('items', [])

    index = build_name_index(new_root)
    fixed = 0
    for it in items:
        name = Path(it['img_path']).name
        if name in index:
            it['img_path'] = str(index[name])
            fixed += 1
    out = {'items': items}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Rebased {fixed}/{len(items)} items to {new_root}. Saved to {out_path}')


if __name__ == '__main__':
    main()
