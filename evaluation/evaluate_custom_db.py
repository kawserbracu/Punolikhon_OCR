import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys
import cv2
import numpy as np

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detection_mnv3seg import MNV3Seg
from evaluation.metrics import calculate_map, calculate_precision_recall_f1, match_boxes, calculate_iou


def load_items(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding='utf-8'))
    return data.get('items', [])


def infer_boxes(model: MNV3Seg, img_path: str, mask_thresh: float = 0.35) -> List[List[float]]:
    img = cv2.imread(img_path)
    if img is None:
        return []
    H, W = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (1024, 768), interpolation=cv2.INTER_AREA)
    import torch
    with torch.no_grad():
        x = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = cv2.resize(probs, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_bin = (mask >= float(mask_thresh)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[List[float]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        x1 = x / W; y1 = y / H; x2 = (x + w) / W; y2 = (y + h) / H
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
    return boxes


def main():
    ap = argparse.ArgumentParser(description='Evaluate custom MobileNetV3 DB-like detector on test set')
    ap.add_argument('--model', type=str, required=True, help='Path to model_best.pt')
    ap.add_argument('--backbone', type=str, default='large', choices=['large','small','hybrid'])
    ap.add_argument('--test_data', type=str, required=True, help='Path to detection_test.json')
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--mask_threshold', type=float, default=0.35)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    import torch
    state = torch.load(args.model, map_location='cpu')
    bb = state.get('backbone', args.backbone)
    model = MNV3Seg(backbone=bb, pretrained=False)
    model.load_state_dict(state['state_dict'])
    model.eval()

    items = load_items(Path(args.test_data))

    all_preds: List[List[List[float]]] = []
    all_gts: List[List[List[float]]] = []

    for it in items:
        img_path = it['img_path']
        preds = infer_boxes(model, img_path, args.mask_threshold)
        gts = it.get('boxes', [])
        all_preds.append(preds)
        all_gts.append(gts)

    # Aggregate metrics at IoU thresholds 0.5 and 0.75 (micro-averaged)
    import numpy as np
    def micro_stats(thr: float):
        tp=fp=fn=0
        ious=[]
        for preds, gts in zip(all_preds, all_gts):
            matches = match_boxes(preds, gts, thr)
            tp += len(matches)
            fp += max(0, len(preds) - len(matches))
            fn += max(0, len(gts) - len(matches))
            ious.extend([m[2] for m in matches])
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        fpr = fp / (tp + fp + 1e-9)
        avg_iou = float(np.mean(ious)) if ious else 0.0
        return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1, fpr=fpr, avg_iou=avg_iou)

    stats_05 = micro_stats(0.5)
    stats_075 = micro_stats(0.75)

    mAP05 = calculate_map(all_preds, all_gts, iou_thresholds=[0.5])
    mAP075 = calculate_map(all_preds, all_gts, iou_thresholds=[0.75])

    results = {
        'mAP@0.5': mAP05,
        'mAP@0.75': mAP075,
        'Precision@0.5': float(stats_05['precision']),
        'Recall@0.5': float(stats_05['recall']),
        'F1@0.5': float(stats_05['f1']),
        'FPR@0.5': float(stats_05['fpr']),
        'AvgIoU@matched@0.5': float(stats_05['avg_iou']),
        'Precision@0.75': float(stats_075['precision']),
        'Recall@0.75': float(stats_075['recall']),
        'F1@0.75': float(stats_075['f1']),
        'FPR@0.75': float(stats_075['fpr']),
        'AvgIoU@matched@0.75': float(stats_075['avg_iou']),
        'mask_threshold': args.mask_threshold,
        'backbone': bb,
        'num_images': len(all_preds),
        'sum_tp@0.5': int(stats_05['tp']),
        'sum_fp@0.5': int(stats_05['fp']),
        'sum_fn@0.5': int(stats_05['fn']),
        'sum_tp@0.75': int(stats_075['tp']),
        'sum_fp@0.75': int(stats_075['fp']),
        'sum_fn@0.75': int(stats_075['fn']),
    }

    # Per-image metrics CSV (at IoU=0.5)
    per_img_csv = out_dir / 'per_image_metrics_0p5.csv'
    with per_img_csv.open('w', encoding='utf-8') as f:
        f.write('img_path,tp,fp,fn,precision,recall,f1,avg_iou\n')
        for it, preds, gts in zip(items, all_preds, all_gts):
            matches = match_boxes(preds, gts, 0.5)
            tp = len(matches)
            fp = max(0, len(preds) - tp)
            fn = max(0, len(gts) - tp)
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            avg_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0
            f.write(f"{it['img_path']},{tp},{fp},{fn},{prec},{rec},{f1},{avg_iou}\n")

    # Summary JSON/CSV
    (out_dir / 'detection_eval_custom.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (out_dir / 'detection_eval_custom.csv').open('w', encoding='utf-8') as f:
        f.write('mAP@0.5,mAP@0.75,Precision@0.5,Recall@0.5,F1@0.5,FPR@0.5,AvgIoU@matched@0.5,Precision@0.75,Recall@0.75,F1@0.75,FPR@0.75,AvgIoU@matched@0.75,mask_threshold,backbone,num_images,sum_tp@0.5,sum_fp@0.5,sum_fn@0.5,sum_tp@0.75,sum_fp@0.75,sum_fn@0.75\n')
        f.write(f"{results['mAP@0.5']},{results['mAP@0.75']},{results['Precision@0.5']},{results['Recall@0.5']},{results['F1@0.5']},{results['FPR@0.5']},{results['AvgIoU@matched@0.5']},{results['Precision@0.75']},{results['Recall@0.75']},{results['F1@0.75']},{results['FPR@0.75']},{results['AvgIoU@matched@0.75']},{results['mask_threshold']},{results['backbone']},{results['num_images']},{results['sum_tp@0.5']},{results['sum_fp@0.5']},{results['sum_fn@0.5']},{results['sum_tp@0.75']},{results['sum_fp@0.75']},{results['sum_fn@0.75']}\n")

    print('Saved custom DB detector evaluation to', out_dir)


if __name__ == '__main__':
    main()
