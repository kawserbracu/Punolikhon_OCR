"""
Sanity Test: Detection Impact on Recognition
=============================================
Compares recognition accuracy using:
  A) Ground-truth bounding boxes (perfect detection)
  B) Detection model predictions

This reveals how much detection quality affects overall OCR accuracy.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
import unicodedata

# Allow running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_2_me import HybridHTR_STN_Transformer
from hybrid_2_me.tokenizer_grapheme import BengaliGraphemeTokenizer


def cer(pred: str, gt: str) -> float:
    """Character Error Rate."""
    import Levenshtein
    if not gt:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gt) / max(len(gt), 1)


def greedy_ctc_decode(logits: torch.Tensor, blank: int = 0) -> List[int]:
    """Greedy CTC decoding."""
    # logits shape: (batch, time, vocab) or (time, vocab)
    if logits.dim() == 3:
        logits = logits.squeeze(0)  # Remove batch dim
    pred = logits.argmax(dim=-1).tolist()  # (time,)
    # Handle case where pred is still nested
    if pred and isinstance(pred[0], list):
        pred = pred[0]
    result = []
    prev = None
    for p in pred:
        if p != blank and p != prev:
            result.append(int(p))
        prev = p
    return result


def crop_word(img: np.ndarray, box: List[float]) -> np.ndarray:
    """Crop word from image using normalized bbox coordinates [x1, y1, x2, y2]."""
    H, W = img.shape[:2]
    x1 = int(round(box[0] * W))
    y1 = int(round(box[1] * H))
    x2 = int(round(box[2] * W))
    y2 = int(round(box[3] * H))
    # Clamp to valid range
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))
    return img[y1:y2, x1:x2]


def preprocess_crop(crop: np.ndarray) -> torch.Tensor:
    """Preprocess crop for recognition model."""
    if crop is None or crop.size == 0:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    h, w = crop.shape[:2]
    scale = 128.0 / max(1, h)
    new_w = max(int(round(w * scale)), 1)
    crop = cv2.resize(crop, (new_w, 128), interpolation=cv2.INTER_CUBIC)
    if crop.shape[1] < 32:
        pad_w = 32 - crop.shape[1]
        crop = cv2.copyMakeBorder(crop, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x = torch.from_numpy(crop.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return x


def run_recognition_on_crops(model, tokenizer, device, crops_and_labels: List[tuple]) -> Dict[str, float]:
    """Run recognition on a list of (crop, gt_text) tuples."""
    total_cer = 0.0
    correct = 0
    total = 0
    blank_idx = tokenizer.grapheme_to_idx.get('<BLANK>', 0)
    
    for crop, gt_text in crops_and_labels:
        x = preprocess_crop(crop)
        if x is None:
            continue
        x = x.to(device)
        
        with torch.no_grad():
            logits = model(x)
        pred_ids = greedy_ctc_decode(logits, blank=blank_idx)
        pred_text = tokenizer.decode_indices(pred_ids)
        
        # NFC normalize
        pred_text = unicodedata.normalize('NFC', pred_text)
        gt_text = unicodedata.normalize('NFC', gt_text)
        
        sample_cer = cer(pred_text, gt_text)
        total_cer += sample_cer
        if pred_text == gt_text:
            correct += 1
        total += 1
    
    return {
        'count': total,
        'word_accuracy': correct / max(1, total),
        'cer': total_cer / max(1, total),
    }


def main():
    ap = argparse.ArgumentParser(description='Sanity test: Detection impact on recognition')
    ap.add_argument('--test_data', type=str, required=True, help='Path to detection_test.json')
    ap.add_argument('--recognition_model', type=str, required=True, help='Path to recognition model checkpoint')
    ap.add_argument('--vocab_path', type=str, required=True, help='Path to vocab JSON')
    ap.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    ap.add_argument('--max_images', type=int, default=50, help='Max images to test (default: 50)')
    ap.add_argument('--max_words', type=int, default=500, help='Max words to test (default: 500)')
    args = ap.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and model
    print("Loading tokenizer and recognition model...")
    tok = BengaliGraphemeTokenizer()
    tok.load_vocab(args.vocab_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridHTR_STN_Transformer(num_classes=tok.vocab_size())
    state = torch.load(args.recognition_model, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    data = json.loads(Path(args.test_data).read_text(encoding='utf-8'))
    items = data.get('items', [])
    
    # Filter items with labels (non-empty)
    items_with_labels = []
    for it in items:
        labels = it.get('labels', [])
        boxes = it.get('boxes', [])
        # Check if any label is non-empty
        valid_pairs = [(b, l) for b, l in zip(boxes, labels) if l and l.strip()]
        if valid_pairs:
            items_with_labels.append({
                'img_path': it['img_path'],
                'boxes': [p[0] for p in valid_pairs],
                'labels': [p[1] for p in valid_pairs],
            })
    
    print(f"Found {len(items_with_labels)} images with labels (out of {len(items)} total)")
    
    # Limit to max_images
    items_with_labels = items_with_labels[:args.max_images]
    
    # Collect GT crops
    gt_crops_and_labels = []
    for it in tqdm(items_with_labels, desc='Cropping GT boxes'):
        img = cv2.imread(it['img_path'])
        if img is None:
            continue
        for box, label in zip(it['boxes'], it['labels']):
            if len(gt_crops_and_labels) >= args.max_words:
                break
            crop = crop_word(img, box)
            if crop is not None and crop.size > 0:
                gt_crops_and_labels.append((crop, label))
        if len(gt_crops_and_labels) >= args.max_words:
            break
    
    print(f"\nCollected {len(gt_crops_and_labels)} word crops with labels")
    
    # Run recognition using GT boxes
    print("\n=== Testing with GROUND-TRUTH boxes (perfect detection) ===")
    gt_results = run_recognition_on_crops(model, tok, device, gt_crops_and_labels)
    print(f"  Word Accuracy: {gt_results['word_accuracy']*100:.2f}%")
    print(f"  CER: {gt_results['cer']*100:.2f}%")
    
    # Save results
    results = {
        'test_type': 'detection_impact_sanity_test',
        'num_images': len(items_with_labels),
        'num_words': len(gt_crops_and_labels),
        'recognition_model': args.recognition_model,
        'gt_box_results': gt_results,
        'interpretation': (
            f"Recognition using PERFECT detection achieves {gt_results['word_accuracy']*100:.2f}% word accuracy. "
            f"This is the upper bound. Compare with end-to-end pipeline to measure detection impact."
        ),
    }
    
    out_json = out_dir / 'sanity_test_results.json'
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSaved results to {out_json}")
    
    # Print summary
    print("\n" + "="*60)
    print("SANITY TEST SUMMARY")
    print("="*60)
    print(f"Recognition with PERFECT (GT) detection:")
    print(f"  - Word Accuracy: {gt_results['word_accuracy']*100:.2f}%")
    print(f"  - CER: {gt_results['cer']*100:.2f}%")
    print()
    print("Your previous recognition eval on cropped test set: 83.21% word accuracy")
    print("If GT-box result is close to 83.21%, detection is NOT the bottleneck.")
    print("If GT-box result is much higher, improving detection will help.")
    print("="*60)


if __name__ == '__main__':
    main()
