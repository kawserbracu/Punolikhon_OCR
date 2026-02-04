"""
Simplified E2E Evaluation - processes images one at a time with explicit cleanup.
"""
import sys
import gc
import cv2
import json
import torch
import time
import numpy as np
import unicodedata
from pathlib import Path
import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_2_me import HybridHTR_STN_Transformer
from hybrid_2_me.tokenizer_grapheme import BengaliGraphemeTokenizer
from models.detection_mnv3seg import MNV3Seg


def cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gt) / max(len(gt), 1)


def iou(box1, box2) -> float:
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


def simple_ctc_decode(logits, blank=0):
    """Simple CTC decode without potential issues."""
    # Move to CPU first
    logits_cpu = logits.cpu()
    
    # Squeeze if needed
    while logits_cpu.dim() > 2:
        logits_cpu = logits_cpu.squeeze(0)
    
    # Get predictions
    preds = logits_cpu.argmax(dim=-1).numpy().tolist()
    
    if isinstance(preds, int):
        preds = [preds]
    
    # CTC collapse
    result = []
    prev = None
    for p in preds:
        if isinstance(p, list):
            p = p[0] if p else 0
        p = int(p)
        if p != blank and p != prev:
            result.append(p)
        prev = p
    
    return result


def main():
    print("="*60)
    print("SIMPLIFIED E2E EVALUATION")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load detection model
    print("\n1. Loading detection model...")
    det_state = torch.load(
        'stupid_again/manifests_combined/runs/det_hybrid_combined/model_best.pt',
        map_location='cpu', weights_only=False
    )
    backbone = det_state.get('backbone', 'hybrid')
    print(f"   Backbone: {backbone}")
    detector = MNV3Seg(backbone=backbone, pretrained=False)
    detector.load_state_dict(det_state['state_dict'])
    detector.to(device).eval()
    print("   Detection model loaded")
    
    # Load recognition model
    print("\n2. Loading recognition model...")
    tokenizer = BengaliGraphemeTokenizer()
    tokenizer.load_vocab('stupid_again/manifests_combined/vocab_grapheme.json')
    print(f"   Vocab size: {tokenizer.vocab_size()}")
    
    recognizer = HybridHTR_STN_Transformer(num_classes=tokenizer.vocab_size())
    rec_state = torch.load(
        'stupid_again/manifests_combined/runs/rec_hybrid2_grapheme_finetune_english/model_best_cer.pt',
        map_location='cpu', weights_only=False
    )
    if 'state_dict' in rec_state:
        recognizer.load_state_dict(rec_state['state_dict'])
    else:
        recognizer.load_state_dict(rec_state)
    recognizer.to(device).eval()
    blank_idx = tokenizer.grapheme_to_idx.get('<BLANK>', 0)
    print("   Recognition model loaded")
    
    # Load test data
    print("\n3. Loading test data...")
    test_data = json.load(open('stupid_again/manifests_combined/detection_test_labeled_only.json', encoding='utf-8'))
    items = test_data['items'][:5]  # Only 5 images for testing
    print(f"   Processing {len(items)} images")
    
    # Metrics
    all_cer = []
    all_correct = 0
    all_total = 0
    det_f1s = []
    
    print("\n" + "="*60)
    start_time = time.time()
    
    for idx, item in enumerate(items):
        img_path = item['img_path']
        gt_boxes = item.get('boxes', [])
        gt_labels = item.get('labels', [])
        
        print(f"\n[{idx+1}/{len(items)}] {Path(img_path).name}")
        print(f"  GT: {len(gt_boxes)} boxes, {sum(1 for l in gt_labels if l.strip())} labels")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print("  ERROR: Could not load image")
            continue
        
        H, W = img.shape[:2]
        print(f"  Image: {W}x{H}")
        
        # Detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (1024, 768))
        img_t = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        img_t = img_t.to(device)
        
        with torch.no_grad():
            logits = detector(img_t)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        
        del img_t, logits
        torch.cuda.empty_cache()
        
        # Find boxes
        mask = cv2.resize(probs, (W, H))
        mask_bin = (mask >= 0.3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pred_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 5 or h < 5:
                continue
            pred_boxes.append([x/W, y/H, (x+w)/W, (y+h)/H])
        
        print(f"  Detected: {len(pred_boxes)} boxes")
        
        # Match boxes
        matches = match_boxes(pred_boxes, gt_boxes)
        tp = len(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        det_f1s.append(f1)
        print(f"  Detection F1: {f1*100:.1f}%")
        
        # Recognition
        words_done = 0
        for pi, gi, _ in matches:
            gt_text = gt_labels[gi] if gi < len(gt_labels) else ""
            gt_text = unicodedata.normalize('NFC', gt_text) if gt_text else ""
            
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
            
            # Preprocess
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ch, cw = rgb_crop.shape[:2]
            scale = 128.0 / max(1, ch)
            new_w = max(int(round(cw * scale)), 1)
            rgb_crop = cv2.resize(rgb_crop, (new_w, 128))
            
            if rgb_crop.shape[1] < 32:
                pad_w = 32 - rgb_crop.shape[1]
                rgb_crop = cv2.copyMakeBorder(rgb_crop, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            # Inference
            x = torch.from_numpy(rgb_crop.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            x = x.to(device)
            
            with torch.no_grad():
                rec_logits = recognizer(x)
            
            # Decode using simple function
            ids = simple_ctc_decode(rec_logits, blank=blank_idx)
            pred_text = tokenizer.decode_indices(ids)
            pred_text = unicodedata.normalize('NFC', pred_text)
            
            del x, rec_logits
            
            # Metrics
            sample_cer = cer(pred_text, gt_text)
            all_cer.append(sample_cer)
            if pred_text == gt_text:
                all_correct += 1
            all_total += 1
            words_done += 1
        
        print(f"  Recognized: {words_done} words")
        
        # Cleanup
        del img, rgb, probs, mask
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final results
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Time: {total_time:.1f}s ({len(items)} images)")
    print(f"\nDetection:")
    print(f"  Avg F1: {sum(det_f1s)/max(len(det_f1s),1)*100:.2f}%")
    print(f"\nRecognition:")
    print(f"  Words: {all_total}")
    print(f"  Accuracy: {all_correct/max(all_total,1)*100:.2f}%")
    print(f"  CER: {sum(all_cer)/max(len(all_cer),1)*100:.2f}%")
    
    # Save results
    results = {
        'num_images': len(items),
        'detection_f1': sum(det_f1s)/max(len(det_f1s),1),
        'words_evaluated': all_total,
        'word_accuracy': all_correct/max(all_total,1),
        'cer': sum(all_cer)/max(len(all_cer),1),
    }
    
    out_path = Path('stupid_again/manifests_combined/runs/last_test/e2e/simple_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
