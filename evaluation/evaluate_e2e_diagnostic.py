"""
E2E Evaluation - DIAGNOSTIC VERSION
====================================
Prints verbose output for each image to diagnose where it's getting stuck.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import cv2
import numpy as np
import torch
import unicodedata
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_2_me import HybridHTR_STN_Transformer
from hybrid_2_me.tokenizer_grapheme import BengaliGraphemeTokenizer
from models.detection_mnv3seg import MNV3Seg
import Levenshtein


def cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gt) / max(len(gt), 1)


def wer(pred: str, gt: str) -> float:
    pred_words = pred.split()
    gt_words = gt.split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return Levenshtein.distance(pred_words, gt_words) / max(len(gt_words), 1)


def iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-9)


def match_boxes(pred_boxes: List[List[float]], gt_boxes: List[List[float]], 
                iou_thresh: float = 0.5) -> List[Tuple[int, int, float]]:
    matches = []
    used_gt = set()
    for pi, pb in enumerate(pred_boxes):
        best_iou = 0
        best_gi = -1
        for gi, gb in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            cur_iou = iou(pb, gb)
            if cur_iou > best_iou and cur_iou >= iou_thresh:
                best_iou = cur_iou
                best_gi = gi
        if best_gi >= 0:
            matches.append((pi, best_gi, best_iou))
            used_gt.add(best_gi)
    return matches


def is_english(text: str) -> bool:
    if not text:
        return True
    ascii_chars = sum(1 for c in text if ord(c) < 128 and c.isalnum())
    return ascii_chars / max(len(text), 1) > 0.5


def greedy_ctc_decode(logits: torch.Tensor, blank: int = 0) -> Tuple[List[int], float]:
    while logits.dim() > 2:
        logits = logits.squeeze(0)
    
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    
    if preds.dim() > 1:
        preds = preds.squeeze()
    preds = preds.tolist()
    
    if isinstance(preds, int):
        preds = [preds]
    
    result = []
    confs = []
    prev = None
    for t, p in enumerate(preds):
        if isinstance(p, list):
            p = p[0] if p else 0
        p = int(p)
        if p != blank and p != prev:
            result.append(p)
            if t < probs.size(0):
                confs.append(probs[t, p].item())
        prev = p
    
    avg_conf = sum(confs) / max(len(confs), 1) if confs else 0.0
    return result, avg_conf


class E2EPipeline:
    def __init__(self, det_model_path: str, rec_model_path: str, vocab_path: str,
                 det_backbone: str = 'large', mask_thresh: float = 0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DIAG] Using device: {self.device}")
        
        print(f"[DIAG] Loading detection model from {det_model_path}...")
        t0 = time.time()
        det_state = torch.load(det_model_path, map_location='cpu', weights_only=False)
        backbone = det_state.get('backbone', det_backbone)
        print(f"[DIAG] Using backbone: {backbone}")
        self.detector = MNV3Seg(backbone=backbone, pretrained=False)
        self.detector.load_state_dict(det_state['state_dict'])
        self.detector.to(self.device)
        self.detector.eval()
        self.mask_thresh = mask_thresh
        print(f"[DIAG] Detection model loaded in {time.time()-t0:.2f}s")
        
        print(f"[DIAG] Loading recognition model from {rec_model_path}...")
        t0 = time.time()
        self.tokenizer = BengaliGraphemeTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        self.recognizer = HybridHTR_STN_Transformer(num_classes=self.tokenizer.vocab_size())
        rec_state = torch.load(rec_model_path, map_location='cpu', weights_only=False)
        if 'state_dict' in rec_state:
            self.recognizer.load_state_dict(rec_state['state_dict'])
        else:
            self.recognizer.load_state_dict(rec_state)
        self.recognizer.to(self.device)
        self.recognizer.eval()
        print(f"[DIAG] Recognition model loaded in {time.time()-t0:.2f}s")
        
        self.blank_idx = self.tokenizer.grapheme_to_idx.get('<BLANK>', 0)
    
    def detect(self, img: np.ndarray) -> List[List[float]]:
        H, W = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (1024, 768), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        img_t = img_t.to(self.device)
        
        with torch.no_grad():
            logits = self.detector(img_t)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        
        mask = cv2.resize(probs, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask >= self.mask_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 5 or h < 5:
                continue
            boxes.append([x/W, y/H, (x+w)/W, (y+h)/H])
        return boxes
    
    def recognize(self, crop: np.ndarray, debug: bool = False) -> Tuple[str, float]:
        if crop is None or crop.size == 0:
            return "", 0.0
        
        if debug:
            print("      [REC] cvtColor...", end="", flush=True)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = 128.0 / max(1, h)
        new_w = max(int(round(w * scale)), 1)
        
        if debug:
            print(f" resize to ({new_w}, 128)...", end="", flush=True)
        rgb = cv2.resize(rgb, (new_w, 128), interpolation=cv2.INTER_CUBIC)
        
        if rgb.shape[1] < 32:
            pad_w = 32 - rgb.shape[1]
            rgb = cv2.copyMakeBorder(rgb, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        if debug:
            print(f" tensor...", end="", flush=True)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        x = x.to(self.device)
        
        if debug:
            print(f" sync...", end="", flush=True)
        # Force CUDA sync before inference
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        if debug:
            print(f" forward...", end="", flush=True)
        with torch.no_grad():
            logits = self.recognizer(x)
        
        if debug:
            print(f" decode...", end="", flush=True)
        ids, conf = greedy_ctc_decode(logits, blank=self.blank_idx)
        text = self.tokenizer.decode_indices(ids)
        
        if debug:
            print(f" done!", flush=True)
        return unicodedata.normalize('NFC', text), conf
    
    def crop_box(self, img: np.ndarray, box: List[float]) -> np.ndarray:
        H, W = img.shape[:2]
        x1 = max(0, int(round(box[0] * W)))
        y1 = max(0, int(round(box[1] * H)))
        x2 = min(W, int(round(box[2] * W)))
        y2 = min(H, int(round(box[3] * H)))
        return img[y1:y2, x1:x2]


def evaluate_e2e_diagnostic(pipeline: E2EPipeline, test_data_path: str, output_dir: str, max_images: int = 0):
    """Run full E2E evaluation with DIAGNOSTIC output."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[DIAG] Loading test data from {test_data_path}...")
    data = json.loads(Path(test_data_path).read_text(encoding='utf-8'))
    items = data.get('items', [])
    
    if max_images > 0:
        items = items[:max_images]
    
    total_images = len(items)
    print(f"[DIAG] Total images to process: {total_images}")
    print("="*70)
    
    det_precisions, det_recalls, det_f1s = [], [], []
    all_cer, all_wer, all_conf = [], [], []
    all_correct, all_total = 0, 0
    bn_cer, bn_wer, bn_conf = [], [], []
    bn_correct, bn_total = 0, 0
    en_cer, en_wer, en_conf = [], [], []
    en_correct, en_total = 0, 0
    
    start_time = time.time()
    
    for idx, item in enumerate(items):
        img_start = time.time()
        img_path = item.get('img_path')
        gt_boxes = item.get('boxes', [])
        gt_labels = item.get('labels', [])
        
        print(f"\n[{idx+1}/{total_images}] Processing: {Path(img_path).name}")
        print(f"  GT boxes: {len(gt_boxes)}")
        sys.stdout.flush()  # Force immediate output
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ERROR: Could not load image!")
            continue
        
        print(f"  Image loaded: {img.shape}")
        sys.stdout.flush()
        
        # Detection
        det_start = time.time()
        pred_boxes = pipeline.detect(img)
        det_time = time.time() - det_start
        print(f"  Detection: {len(pred_boxes)} boxes found ({det_time:.2f}s)")
        sys.stdout.flush()
        
        # Detection metrics
        matches = match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5)
        tp = len(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        
        det_precisions.append(precision)
        det_recalls.append(recall)
        det_f1s.append(f1)
        
        print(f"  Matches: {tp}, Det P/R/F1: {precision:.2f}/{recall:.2f}/{f1:.2f}")
        sys.stdout.flush()
        
        # Recognition on matched boxes
        rec_start = time.time()
        words_processed = 0
        print(f"  Starting recognition on {len(matches)} matches...")
        sys.stdout.flush()
        
        for mi, (pi, gi, _) in enumerate(matches):
            gt_text = gt_labels[gi] if gi < len(gt_labels) else ""
            gt_text = unicodedata.normalize('NFC', gt_text) if gt_text else ""
            
            if not gt_text.strip():
                continue
            
            # Debug first few
            if words_processed < 3:
                print(f"    Word {words_processed}: cropping box {pi}...")
                sys.stdout.flush()
            
            crop = pipeline.crop_box(img, pred_boxes[pi])
            
            if words_processed < 3:
                print(f"    Word {words_processed}: crop shape {crop.shape if crop is not None else 'None'}, recognizing...")
                sys.stdout.flush()
            
            pred_text, conf = pipeline.recognize(crop, debug=(words_processed < 3))
            
            if words_processed < 3:
                print(f"    Word {words_processed}: recognized, CER calculating...")
                sys.stdout.flush()
            
            words_processed += 1
            
            sample_cer = cer(pred_text, gt_text)
            sample_wer = wer(pred_text, gt_text)
            exact_match = 1 if pred_text == gt_text else 0
            
            all_cer.append(sample_cer)
            all_wer.append(sample_wer)
            all_conf.append(conf)
            all_correct += exact_match
            all_total += 1
            
            if is_english(gt_text):
                en_cer.append(sample_cer)
                en_wer.append(sample_wer)
                en_conf.append(conf)
                en_correct += exact_match
                en_total += 1
            else:
                bn_cer.append(sample_cer)
                bn_wer.append(sample_wer)
                bn_conf.append(conf)
                bn_correct += exact_match
                bn_total += 1
        
        rec_time = time.time() - rec_start
        img_time = time.time() - img_start
        
        print(f"  Recognition: {words_processed} words ({rec_time:.2f}s)")
        print(f"  Total image time: {img_time:.2f}s")
        
        # Running stats
        elapsed = time.time() - start_time
        avg_per_img = elapsed / (idx + 1)
        remaining = avg_per_img * (total_images - idx - 1)
        print(f"  Running: {all_total} words, Acc={all_correct/max(all_total,1)*100:.1f}%, "
              f"CER={sum(all_cer)/max(len(all_cer),1)*100:.1f}%")
        print(f"  ETA: {remaining/60:.1f} min remaining")
        sys.stdout.flush()
    
    # Compute aggregates
    results = {
        'num_images': len(items),
        'detection': {
            'precision': sum(det_precisions) / max(len(det_precisions), 1),
            'recall': sum(det_recalls) / max(len(det_recalls), 1),
            'f1': sum(det_f1s) / max(len(det_f1s), 1),
        },
        'overall': {
            'num_words_evaluated': all_total,
            'word_accuracy': all_correct / max(all_total, 1),
            'cer': sum(all_cer) / max(len(all_cer), 1),
            'wer': sum(all_wer) / max(len(all_wer), 1),
            'avg_confidence': sum(all_conf) / max(len(all_conf), 1),
        },
        'bangla': {
            'num_words': bn_total,
            'word_accuracy': bn_correct / max(bn_total, 1),
            'cer': sum(bn_cer) / max(len(bn_cer), 1),
            'wer': sum(bn_wer) / max(len(bn_wer), 1),
            'avg_confidence': sum(bn_conf) / max(len(bn_conf), 1),
        },
        'english': {
            'num_words': en_total,
            'word_accuracy': en_correct / max(en_total, 1),
            'cer': sum(en_cer) / max(len(en_cer), 1),
            'wer': sum(en_wer) / max(len(en_wer), 1),
            'avg_confidence': sum(en_conf) / max(len(en_conf), 1),
        },
    }
    
    out_json = out_dir / 'e2e_evaluation.json'
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("END-TO-END EVALUATION COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("="*70)
    
    print("\n[DETECTION METRICS]")
    print(f"  Precision: {results['detection']['precision']*100:.2f}%")
    print(f"  Recall:    {results['detection']['recall']*100:.2f}%")
    print(f"  F1 Score:  {results['detection']['f1']*100:.2f}%")
    
    print("\n[OVERALL RECOGNITION (E2E)]")
    print(f"  Words Evaluated: {results['overall']['num_words_evaluated']}")
    print(f"  Word Accuracy:   {results['overall']['word_accuracy']*100:.2f}%")
    print(f"  CER:             {results['overall']['cer']*100:.2f}%")
    print(f"  WER:             {results['overall']['wer']*100:.2f}%")
    
    print("\n[BANGLA]")
    print(f"  Words: {results['bangla']['num_words']}")
    print(f"  Word Accuracy: {results['bangla']['word_accuracy']*100:.2f}%")
    print(f"  CER: {results['bangla']['cer']*100:.2f}%")
    
    print("\n[ENGLISH]")
    print(f"  Words: {results['english']['num_words']}")
    print(f"  Word Accuracy: {results['english']['word_accuracy']*100:.2f}%")
    print(f"  CER: {results['english']['cer']*100:.2f}%")
    
    print(f"\nResults saved to {out_json}")
    
    return results


def main():
    ap = argparse.ArgumentParser(description='E2E OCR Evaluation - DIAGNOSTIC')
    ap.add_argument('--det_model', required=True, help='Path to detection model checkpoint')
    ap.add_argument('--rec_model', required=True, help='Path to recognition model checkpoint')
    ap.add_argument('--vocab', required=True, help='Path to vocab JSON')
    ap.add_argument('--test_data', required=True, help='Path to detection_test.json')
    ap.add_argument('--output_dir', required=True, help='Output directory')
    ap.add_argument('--max_images', type=int, default=0, help='Max images (0=all)')
    ap.add_argument('--det_backbone', default='large', choices=['large', 'small'])
    ap.add_argument('--mask_thresh', type=float, default=0.3)
    args = ap.parse_args()
    
    pipeline = E2EPipeline(
        det_model_path=args.det_model,
        rec_model_path=args.rec_model,
        vocab_path=args.vocab,
        det_backbone=args.det_backbone,
        mask_thresh=args.mask_thresh
    )
    
    evaluate_e2e_diagnostic(pipeline, args.test_data, args.output_dir, args.max_images)


if __name__ == '__main__':
    main()
