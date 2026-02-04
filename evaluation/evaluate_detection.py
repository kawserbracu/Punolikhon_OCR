import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
from tqdm import tqdm

import sys
from pathlib import Path
# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.detection import build_detection_model
from evaluation.metrics import calculate_map, calculate_precision_recall_f1, calculate_iou


def load_detection_items(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding='utf-8'))
    return data.get('items', [])


def run_inference(model, img_path: str) -> List[List[float]]:
    # Use doctr predictor; returns polygons normally; approximate as boxes
    import numpy as np
    from doctr.io.image import read_img_as_tensor

    page = read_img_as_tensor(img_path)
    preds = model([page])
    # Flatten polygons to boxes (min/max)
    boxes: List[List[float]] = []
    obj = preds[0] if isinstance(preds, (list, tuple)) and len(preds) > 0 else preds
    if hasattr(obj, 'pages'):
        for p in obj.pages:
            for l in getattr(p, 'lines', []):
                for w in getattr(l, 'words', []):
                    poly = np.array(w.geometry, dtype=float)
                    xs = poly[:, 0]; ys = poly[:, 1]
                    boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    elif isinstance(obj, dict):
        for p in obj.get('pages', []):
            for blk in p.get('blocks', []):
                for ln in blk.get('lines', []):
                    for wd in ln.get('words', []):
                        poly = np.array(wd.get('geometry', []), dtype=float)
                        if poly.size == 0:
                            continue
                        xs = poly[:, 0]; ys = poly[:, 1]
                        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    return boxes


def main():
    ap = argparse.ArgumentParser(description='Evaluate detection mAP/Precision/Recall/F1 on test set')
    ap.add_argument('--model_path', type=str, default=None, help='Optional checkpoint path (not used for predictor)')
    ap.add_argument('--test_data', type=str, required=True, help='Path to detection_test.json')
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    test_items = load_detection_items(Path(args.test_data))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_detection_model(pretrained=True)

    all_preds: List[List[List[float]]] = []
    all_gts: List[List[List[float]]] = []
    avg_ious = []

    for it in tqdm(test_items, desc='Eval detection'):
        img_path = it['img_path']
        preds = run_inference(model, img_path)
        gts = it.get('boxes', [])
        all_preds.append(preds)
        all_gts.append(gts)
        # Average IoU (greedy best match)
        if preds and gts:
            from .metrics import match_boxes
            matches = match_boxes(preds, gts, 0.5)
            if matches:
                avg_ious.append(sum(m[2] for m in matches) / len(matches))
            else:
                avg_ious.append(0.0)
        else:
            avg_ious.append(0.0)

    mAP = calculate_map(all_preds, all_gts, iou_thresholds=[0.5])
    P, R, F1 = 0.0, 0.0, 0.0
    ps, rs, f1s = [], [], []
    for preds, gts in zip(all_preds, all_gts):
        p, r, f1 = calculate_precision_recall_f1(preds, gts, 0.5)
        ps.append(p); rs.append(r); f1s.append(f1)
    if ps:
        P, R, F1 = float(sum(ps)/len(ps)), float(sum(rs)/len(rs)), float(sum(f1s)/len(f1s))

    results = {
        'mAP@0.5': mAP,
        'Precision@0.5': P,
        'Recall@0.5': R,
        'F1@0.5': F1,
        'AverageIoU': float(sum(avg_ious)/len(avg_ious) if avg_ious else 0.0),
    }

    (out_dir / 'detection_eval.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    # CSV table with a single row
    with (out_dir / 'detection_eval.csv').open('w', encoding='utf-8') as f:
        f.write('mAP@0.5,Precision@0.5,Recall@0.5,F1@0.5,AverageIoU\n')
        f.write(f"{results['mAP@0.5']},{results['Precision@0.5']},{results['Recall@0.5']},{results['F1@0.5']},{results['AverageIoU']}\n")

    print('Saved detection evaluation to', out_dir)


if __name__ == '__main__':
    main()
