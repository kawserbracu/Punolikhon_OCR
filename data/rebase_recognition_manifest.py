import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def rebase_crop_path(p: str, old_root: Path | None, new_root: Path | None) -> str:
    path = Path(p)

    if path.is_absolute() and old_root is not None:
        try:
            rel = path.relative_to(old_root)
            if new_root is None:
                return rel.as_posix()
            return (new_root / rel).as_posix()
        except Exception:
            pass

    if old_root is not None:
        try:
            rel = Path(p).resolve().relative_to(old_root)
            if new_root is None:
                return rel.as_posix()
            return (new_root / rel).as_posix()
        except Exception:
            pass

    s = str(p).replace("\\", "/")
    marker = "crops_3x"
    if marker in s:
        tail = s.split(marker, 1)[1].lstrip("/")
        if new_root is None:
            return f"{marker}/{tail}"
        return (new_root / tail).as_posix()

    return s


def main():
    ap = argparse.ArgumentParser(description="Rebase recognition manifest crop_path entries (absolute -> relative or new root)")
    ap.add_argument("--input_json", type=str, required=True)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--old_root", type=str, default="", help="Old crops root to strip (e.g., dataset/crops_3x_combined)")
    ap.add_argument("--new_root", type=str, default="", help="New crops root to prefix (e.g., crops_3x_combined)")
    args = ap.parse_args()

    in_path = Path(args.input_json)
    out_path = Path(args.output_json)

    old_root = Path(args.old_root).resolve() if args.old_root else None
    new_root = Path(args.new_root) if args.new_root else None

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of items")

    out: List[Dict[str, Any]] = []
    fixed = 0
    for it in data:
        if not isinstance(it, dict):
            continue
        if "crop_path" in it:
            before = it["crop_path"]
            after = rebase_crop_path(str(before), old_root, new_root)
            if after != before:
                fixed += 1
            it["crop_path"] = after
        out.append(it)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rebased {fixed}/{len(out)} items. Saved to {out_path}")


if __name__ == "__main__":
    main()
