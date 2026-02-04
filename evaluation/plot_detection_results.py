import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import csv


def load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description='Plot detection results (mAP/Precision/Recall/F1) from multiple eval jsons')
    ap.add_argument('--det_eval', type=str, nargs='+', required=True,
                    help='One or more entries like label=path_to_detection_eval.json')
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: List[str] = []
    rows: List[Dict[str, Any]] = []
    metrics = ['mAP@0.5', 'Precision@0.5', 'Recall@0.5', 'F1@0.5']

    for entry in args.det_eval:
        if '=' not in entry:
            continue
        label, path = entry.split('=', 1)
        data = load_json(Path(path))
        if not data:
            continue
        labels.append(label)
        row = {m: float(data.get(m, 0.0)) for m in metrics}
        row['label'] = label
        rows.append(row)

    if not rows:
        print('No valid detection evals found.')
        return

    # Save combined CSV
    csv_path = out_dir / 'detection_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + metrics)
        for r in rows:
            writer.writerow([r['label']] + [r[m] for m in metrics])

    # Plot bars per metric
    x = np.arange(len(rows))
    for m in metrics:
        vals = [r[m] for r in rows]
        plt.figure(figsize=(8, 4))
        plt.bar(x, vals, color='#4C78A8')
        plt.xticks(x, labels, rotation=20, ha='right')
        plt.ylabel(m)
        plt.title(f'Detection: {m}')
        plt.tight_layout()
        plt.savefig(out_dir / f'detection_{m.replace("@","_at_").replace("/","-")}.png', dpi=150)
        plt.close()

    # Combined grouped bar (Precision/Recall/F1)
    group_metrics = ['Precision@0.5', 'Recall@0.5', 'F1@0.5']
    width = 0.22
    plt.figure(figsize=(9, 4))
    for i, gm in enumerate(group_metrics):
        vals = [r[gm] for r in rows]
        plt.bar(x + (i - 1)*width, vals, width=width, label=gm)
    plt.xticks(x, labels, rotation=20, ha='right')
    plt.ylabel('Score')
    plt.title('Detection: Precision/Recall/F1 at IoU=0.5')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'detection_prf_grouped.png', dpi=150)
    plt.close()

    print('Saved detection plots and CSV to', out_dir)


if __name__ == '__main__':
    main()
