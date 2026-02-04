import argparse
import json
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import csv


def load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description='Plot recognition results (CER/WER/Accuracy)')
    ap.add_argument('--rec_eval', type=str, nargs='+', required=True,
                    help='One or more entries like label=path_to_recognition_eval.json')
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    rows = []
    metrics = ['word_accuracy', 'CER_mean', 'WER_mean']

    for entry in args.rec_eval:
        if '=' not in entry:
            continue
        label, path = entry.split('=', 1)
        data = load_json(Path(path))
        if not data:
            continue
        labels.append(label)
        rows.append({m: float(data.get(m, 0.0)) for m in metrics})

    if not rows:
        print('No valid recognition evals found.')
        return

    # Save combined CSV
    with (out_dir / 'recognition_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + metrics)
        for label, row in zip(labels, rows):
            writer.writerow([label] + [row[m] for m in metrics])

    # Plot bars per metric
    import numpy as np
    x = np.arange(len(rows))
    colors = ['#72B7B2', '#E45756', '#F58518']

    for i, m in enumerate(metrics):
        vals = [row[m] for row in rows]
        plt.figure(figsize=(8, 4))
        plt.bar(x, vals, color=colors[i % len(colors)])
        plt.xticks(x, labels, rotation=20, ha='right')
        plt.ylabel(m)
        plt.title(f'Recognition: {m}')
        plt.tight_layout()
        plt.savefig(out_dir / f'recognition_{m}.png', dpi=150)
        plt.close()

    # Grouped bar (Accuracy vs CER/WER) – note lower is better for CER/WER
    width = 0.25
    plt.figure(figsize=(9, 4))
    acc = [row['word_accuracy'] for row in rows]
    cer = [row['CER_mean'] for row in rows]
    wer = [row['WER_mean'] for row in rows]
    plt.bar(x - width, acc, width=width, label='Accuracy', color='#59A14F')
    plt.bar(x, cer, width=width, label='CER (lower is better)', color='#EDC948')
    plt.bar(x + width, wer, width=width, label='WER (lower is better)', color='#B07AA1')
    plt.xticks(x, labels, rotation=20, ha='right')
    plt.ylabel('Score')
    plt.title('Recognition: Accuracy vs CER/WER')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'recognition_grouped.png', dpi=150)
    plt.close()

    print('Saved recognition plots and CSV to', out_dir)


if __name__ == '__main__':
    main()
