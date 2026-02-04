import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def count_items(p: Path) -> int:
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        return len(data.get('items', []))
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description='Plot dataset split sizes from detection/recognition manifests')
    ap.add_argument('--data_dir', type=str, required=True, help='Folder containing detection_*.json and/or recognition_*.json')
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detection
    det_counts = {
        'train': count_items(data_dir / 'detection_train.json'),
        'val': count_items(data_dir / 'detection_val.json'),
        'test': count_items(data_dir / 'detection_test.json'),
    }

    # Recognition
    rec_counts = {
        'train': count_items(data_dir / 'recognition_train.json'),
        'val': count_items(data_dir / 'recognition_val.json'),
        'test': count_items(data_dir / 'recognition_test.json'),
    }

    # Plot
    import numpy as np
    x = np.arange(3)
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, [det_counts['train'], det_counts['val'], det_counts['test']], width=width, label='Detection')
    plt.bar(x + width/2, [rec_counts['train'], rec_counts['val'], rec_counts['test']], width=width, label='Recognition')
    plt.xticks(x, ['Train', 'Val', 'Test'])
    plt.ylabel('Items')
    plt.title('Dataset Split Sizes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'dataset_splits.png', dpi=150)
    plt.close()

    # Save CSV
    with (out_dir / 'dataset_splits.csv').open('w', encoding='utf-8') as f:
        f.write('split,det_count,rec_count\n')
        for k in ['train','val','test']:
            f.write(f"{k},{det_counts[k]},{rec_counts[k]}\n")

    print('Saved dataset split plot and CSV to', out_dir)


if __name__ == '__main__':
    main()
