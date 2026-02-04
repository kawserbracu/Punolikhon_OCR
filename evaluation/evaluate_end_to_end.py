import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from ..inference.ocr_pipeline import EndToEndOCR
from ..evaluation.metrics import match_boxes, calculate_precision_recall_f1, calculate_cer, calculate_wer


def load_detection_items(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding='utf-8'))
    return data.get('items', [])


def run_e2e_on_items(pipeline: EndToEndOCR, items: List[Dict[str, Any]]) -> Dict[str, float]:
    total_words = 0
    det_ps, det_rs, det_f1s = [], [], []
    e2e_correct = 0
    cer_vals, wer_vals = [], []

    for it in tqdm(items, desc='E2E eval'):
        img_path = it['img_path']
        gts = it.get('boxes', [])
        gt_texts = it.get('labels', []) or [""] * len(gts)
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        # pipeline detection
        det_boxes = pipeline.detect_words(img)
        # match boxes
        matches = match_boxes(det_boxes, gts, iou_threshold=0.5)
        # detection metrics per image
        p, r, f1 = calculate_precision_recall_f1(det_boxes, gts, 0.5)
        det_ps.append(p); det_rs.append(r); det_f1s.append(f1)

        # recognize matched words
        for pi, gi, _ in matches:
            db = det_boxes[pi]
            x1 = int(round(db[0] * W)); y1 = int(round(db[1] * H))
            x2 = int(round(db[2] * W)); y2 = int(round(db[3] * H))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pred_text = pipeline.recognize_word(crop)
            gt_text = gt_texts[gi] if gi < len(gt_texts) else ""
            total_words += 1
            if pred_text == gt_text:
                e2e_correct += 1
            cer_vals.append(calculate_cer(pred_text, gt_text))
            wer_vals.append(calculate_wer(pred_text, gt_text))

    results = {
        'Detection_P@0.5': float(np.mean(det_ps) if det_ps else 0.0),
        'Detection_R@0.5': float(np.mean(det_rs) if det_rs else 0.0),
        'Detection_F1@0.5': float(np.mean(det_f1s) if det_f1s else 0.0),
        'E2E_Word_Accuracy': float(e2e_correct / max(1, total_words)),
        'E2E_CER': float(np.mean(cer_vals) if cer_vals else 0.0),
        'E2E_WER': float(np.mean(wer_vals) if wer_vals else 0.0),
    }
    return results


def main():
    ap = argparse.ArgumentParser(description='Evaluate End-to-End OCR with 9 model combinations (3 det x 3 rec)')
    ap.add_argument('--test_data', type=str, required=True, help='Path to detection_test.json (includes boxes+labels)')
    ap.add_argument('--models_dir', type=str, required=True, help='Directory containing 3 detection and 3 recognition models')
    ap.add_argument('--vocab_path', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    items = load_detection_items(Path(args.test_data))
    models_dir = Path(args.models_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expect files named det1.pt det2.pt det3.pt and rec1.pt rec2.pt rec3.pt (recognition). Detection path not actually loaded.
    det_paths = sorted([p for p in models_dir.glob('det*.pt')])
    rec_paths = sorted([p for p in models_dir.glob('rec*.pt')])
    if not det_paths:
        det_paths = [None, None, None]
    if not rec_paths:
        raise RuntimeError('No recognition models found (expected rec*.pt)')

    rows: List[Dict[str, Any]] = []
    for di, dpath in enumerate(det_paths[:3]):
        for ri, rpath in enumerate(rec_paths[:3]):
            name = f'det{di+1}_rec{ri+1}'
            pipeline = EndToEndOCR(detection_model_path=str(dpath) if dpath else None,
                                   recognition_model_path=str(rpath),
                                   tokenizer_path=args.vocab_path)
            metrics = run_e2e_on_items(pipeline, items)
            row = {'combo': name, **metrics}
            rows.append(row)

    # Save CSV and JSON
    import csv
    csv_path = out_dir / 'e2e_comparison.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (out_dir / 'e2e_comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('Saved E2E comparison to', out_dir)


if __name__ == '__main__':
    import numpy as np
    main()
