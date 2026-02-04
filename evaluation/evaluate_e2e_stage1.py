"""
Stage 1: Detection - Run detection and export crops with metadata.
"""
import sys
import cv2
import json
import torch
import time
import numpy as np
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detection_mnv3seg import MNV3Seg
from baocr_paths import load_paths_from_md


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-9)


def match_boxes(pred_boxes, gt_boxes, thresh=0.5):
    matches = []
    used_gt = set()
    for pi, pb in enumerate(pred_boxes):
        best_iou, best_gi = 0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            cur_iou = iou(pb, gb)
            if cur_iou > best_iou and cur_iou >= thresh:
                best_iou = cur_iou
                best_gi = gi
        if best_gi >= 0:
            matches.append((pi, best_gi, best_iou))
            used_gt.add(best_gi)
    return matches


def main():
    ap = argparse.ArgumentParser(description='Stage 1 (E2E): detection + crop export')
    ap.add_argument('--paths_md', type=str, default=str(PROJECT_ROOT / 'lassst.md'), help='Path to lassst.md containing JSON paths')
    args = ap.parse_args()

    paths = load_paths_from_md(args.paths_md)

    print("="*60)
    print("STAGE 1: DETECTION & CROP EXPORT")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load detection model
    print("\nLoading detection model...")
    det_ckpt = paths.get_nested('detection', 'model_checkpoint', default=None)
    if not det_ckpt:
        raise ValueError('Missing detection.model_checkpoint in lassst.md')
    det_state = torch.load(det_ckpt, map_location='cpu', weights_only=False)
    backbone = det_state.get('backbone', 'hybrid')
    print(f"Backbone: {backbone}")
    
    detector = MNV3Seg(backbone=backbone, pretrained=False)
    detector.load_state_dict(det_state['state_dict'])
    detector.to(device).eval()
    print("Detection model loaded")
    
    # Load test data
    print("\nLoading test data...")
    test_json = paths.get_nested('detection', 'test_labeled_json', default=None)
    if not test_json:
        # Fallback to configured test_json if a labeled-only split is not provided
        test_json = paths.get_nested('detection', 'test_json', default=None)
    if not test_json:
        raise ValueError('Missing detection.test_json (or detection.test_labeled_json) in lassst.md')
    test_data = json.load(open(test_json, encoding='utf-8'))
    items = test_data['items']
    print(f"Processing {len(items)} images")
    
    # Output directory for crops
    out_dir = Path(paths.get_nested('e2e', 'output_dir', default=str(PROJECT_ROOT / 'runs' / 'last_test'))) / 'e2e_crops'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Results for stage 2
    crop_manifest = []
    det_metrics = {'precisions': [], 'recalls': [], 'f1s': []}
    
    print("\n" + "="*60)
    start_time = time.time()
    
    for idx, item in enumerate(items):
        img_path = item['img_path']
        gt_boxes = item.get('boxes', [])
        gt_labels = item.get('labels', [])
        
        print(f"[{idx+1}/{len(items)}] {Path(img_path).name}", end=" ")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print("SKIP (no image)")
            continue
        
        H, W = img.shape[:2]
        
        # Detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (1024, 768))
        img_t = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        img_t = img_t.to(device)
        
        with torch.no_grad():
            logits = detector(img_t)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        
        # Find boxes
        mask = cv2.resize(probs, (W, H))
        mask_threshold = float(paths.get_nested('detection', 'mask_threshold', default=0.3))
        mask_bin = (mask >= mask_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pred_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 5 or h < 5:
                continue
            pred_boxes.append([x/W, y/H, (x+w)/W, (y+h)/H])
        
        # Match boxes
        matches = match_boxes(pred_boxes, gt_boxes)
        tp = len(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        
        det_metrics['precisions'].append(precision)
        det_metrics['recalls'].append(recall)
        det_metrics['f1s'].append(f1)
        
        # Export crops for matched boxes
        crops_exported = 0
        img_stem = Path(img_path).stem
        
        for mi, (pi, gi, _) in enumerate(matches):
            gt_text = gt_labels[gi] if gi < len(gt_labels) else ""
            
            if not gt_text.strip():
                continue
            
            # Crop
            x1 = max(0, int(round(pred_boxes[pi][0] * W)))
            y1 = max(0, int(round(pred_boxes[pi][1] * H)))
            x2 = min(W, int(round(pred_boxes[pi][2] * W)))
            y2 = min(H, int(round(pred_boxes[pi][3] * H)))
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Save crop
            crop_name = f"{img_stem}_crop{mi:03d}.jpg"
            crop_path = out_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            
            # Add to manifest
            crop_manifest.append({
                'crop_path': str(crop_path),
                'gt_text': gt_text,
                'source_image': img_path,
                'box_idx': pi
            })
            crops_exported += 1
        
        print(f"F1={f1*100:.0f}% crops={crops_exported}")
        
        # Cleanup
        del img_t, logits, probs, mask
        torch.cuda.empty_cache()
    
    # Save results
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("DETECTION COMPLETE")
    print("="*60)
    print(f"Time: {total_time:.1f}s")
    print(f"Images: {len(items)}")
    print(f"Crops exported: {len(crop_manifest)}")
    print(f"Avg F1: {sum(det_metrics['f1s'])/max(len(det_metrics['f1s']),1)*100:.2f}%")
    
    # Save manifest for stage 2
    manifest_path = out_dir / 'crop_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            'crops': crop_manifest,
            'detection_metrics': {
                'avg_precision': sum(det_metrics['precisions'])/max(len(det_metrics['precisions']),1),
                'avg_recall': sum(det_metrics['recalls'])/max(len(det_metrics['recalls']),1),
                'avg_f1': sum(det_metrics['f1s'])/max(len(det_metrics['f1s']),1),
            },
            'num_images': len(items),
            'time_seconds': total_time
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nManifest saved to: {manifest_path}")
    print("Run evaluate_e2e_stage2.py next for recognition")


if __name__ == '__main__':
    main()
