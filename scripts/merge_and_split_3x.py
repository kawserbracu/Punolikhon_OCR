import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def canonical_image_id_from_crop(item: dict) -> str:
    p = Path(item["crop_path"])
    stem = p.stem
    if "_word" in stem:
        orig_stem = stem.rsplit("_word", 1)[0]
    elif "_w" in stem and stem.rsplit("_w", 1)[1].isdigit():
        orig_stem = stem.rsplit("_w", 1)[0]
    else:
        orig_stem = stem
    source = item.get("source", "unknown")
    return f"{source}/{orig_stem}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input recognition manifest files (list of crop items)")
    parser.add_argument("--output_dir", required=True, help="Output directory for recognition_train/val/test.json")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1")

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_items = []
    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
            all_items.extend(items)

    items_by_image = defaultdict(list)
    for item in all_items:
        img_id = canonical_image_id_from_crop(item)
        items_by_image[img_id].append(item)

    img_ids = list(items_by_image.keys())
    random.shuffle(img_ids)

    n = len(img_ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_ids = set(img_ids[:n_train])
    val_ids = set(img_ids[n_train : n_train + n_val])
    test_ids = set(img_ids[n_train + n_val :])

    train_items, val_items, test_items = [], [], []
    for img_id, items in items_by_image.items():
        if img_id in train_ids:
            train_items.extend(items)
        elif img_id in val_ids:
            val_items.extend(items)
        else:
            test_items.extend(items)

    def save(name: str, data: list):
        p = output_dir / f"recognition_{name}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {p} ({len(data)} items)")

    save("train", train_items)
    save("val", val_items)
    save("test", test_items)


if __name__ == "__main__":
    main()
