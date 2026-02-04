import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser(description='Aggregate training logs and evaluation outputs into tables/plots')
    ap.add_argument('--logs_dir', type=str, required=True)
    ap.add_argument('--eval_dir', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Detection comparison (if multiple runs exist)
    det_rows: List[Dict[str, Any]] = []
    for p in eval_dir.glob('**/detection_eval.json'):
        name = p.parent.name
        det = load_json(p)
        det_rows.append({'run': name, **det})
    if det_rows:
        df_det = pd.DataFrame(det_rows)
        df_det.to_csv(out_dir / 'detection_comparison.csv', index=False)

    # Recognition comparison
    rec_rows: List[Dict[str, Any]] = []
    for p in eval_dir.glob('**/recognition_eval.json'):
        name = p.parent.name
        rec = load_json(p)
        rec_rows.append({'run': name, **rec})
    if rec_rows:
        df_rec = pd.DataFrame(rec_rows)
        df_rec.to_csv(out_dir / 'recognition_comparison.csv', index=False)

    # End-to-end comparison (top 5)
    e2e_path = eval_dir / 'e2e_comparison.json'
    if e2e_path.exists():
        e2e = load_json(e2e_path)
        df_e2e = pd.DataFrame(e2e)
        df_e2e.sort_values(by=['E2E_Word_Accuracy'], ascending=False, inplace=True)
        df_e2e.head(5).to_csv(out_dir / 'e2e_top5.csv', index=False)

    # Plot examples: if training logs are available
    for p in logs_dir.glob('**/training_log.csv'):
        try:
            df = pd.read_csv(p)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(df['epoch'], df['train_loss'], label='train_loss')
            ax.plot(df['epoch'], df['val_loss'], label='val_loss')
            ax.set_title(p.parent.name)
            ax.legend()
            plt.tight_layout()
            fig.savefig(out_dir / f'{p.parent.name}_curves.png', dpi=150)
            plt.close(fig)
        except Exception:
            continue

    print('Generated results under', out_dir)


if __name__ == '__main__':
    main()
