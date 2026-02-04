"""
Stage 2: Recognition (CPU Version) - Runs on CPU to avoid GPU issues.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import json
import torch
import time
import unicodedata
from pathlib import Path
import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_2_me import HybridHTR_STN_Transformer
from hybrid_2_me.tokenizer_grapheme import BengaliGraphemeTokenizer


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


def is_bangla(text):
    if not text:
        return False
    for c in text:
        if '\u0980' <= c <= '\u09FF':
            return True
    return False


def main():
    print("="*60)
    print("STAGE 2: RECOGNITION (CPU)")
    print("="*60)
    
    # Force CPU
    device = torch.device('cpu')
    print(f"Device: {device} (forced CPU mode)")
    
    manifest_path = Path('stupid_again/manifests_combined/runs/last_test/e2e_crops/crop_manifest.json')
    if not manifest_path.exists():
        print("ERROR: Run evaluate_e2e_stage1.py first!")
        return
    
    manifest = json.load(open(manifest_path, encoding='utf-8'))
    crops = manifest['crops']
    print(f"Loaded {len(crops)} crops from stage 1")
    
    print("\nLoading recognition model...")
    tokenizer = BengaliGraphemeTokenizer()
    tokenizer.load_vocab('stupid_again/manifests_combined/vocab_grapheme.json')
    print(f"Vocab size: {tokenizer.vocab_size()}")
    
    recognizer = HybridHTR_STN_Transformer(num_classes=tokenizer.vocab_size())
    rec_state = torch.load(
        'stupid_again/manifests_combined/runs/rec_hybrid2_grapheme_finetune_english/model_best_cer.pt',
        map_location='cpu', weights_only=False
    )
    if 'state_dict' in rec_state:
        recognizer.load_state_dict(rec_state['state_dict'])
    else:
        recognizer.load_state_dict(rec_state)
    recognizer.eval()  # Keep on CPU
    blank_idx = tokenizer.grapheme_to_idx.get('<BLANK>', 0)
    print("Recognition model loaded (CPU)")
    
    print("\n" + "="*60)
    start_time = time.time()
    
    all_cer = []
    all_wer = []
    all_correct = 0
    all_total = 0
    bn_cer, bn_correct, bn_total = [], 0, 0
    en_cer, en_correct, en_total = [], 0, 0
    
    results = []
    
    for i, crop_info in enumerate(crops):
        crop_path = crop_info['crop_path']
        gt_text = unicodedata.normalize('NFC', crop_info['gt_text'])
        
        if (i + 1) % 100 == 0 or i == 0 or i == len(crops) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / max(elapsed, 0.001)
            eta = (len(crops) - i - 1) / max(rate, 0.001)
            print(f"Processing {i+1}/{len(crops)}... ({rate:.1f}/s, ETA: {eta/60:.1f}min)")
        
        crop = cv2.imread(crop_path)
        if crop is None:
            continue
        
        # Preprocess
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = 128.0 / max(1, h)
        new_w = max(int(round(w * scale)), 1)
        rgb = cv2.resize(rgb, (new_w, 128))
        
        if rgb.shape[1] < 32:
            pad_w = 32 - rgb.shape[1]
            rgb = cv2.copyMakeBorder(rgb, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        with torch.no_grad():
            logits = recognizer(x)
        
        # CTC decode
        while logits.dim() > 2:
            logits = logits.squeeze(0)
        preds = logits.argmax(dim=-1).numpy().tolist()
        if isinstance(preds, int):
            preds = [preds]
        
        collapsed = []
        prev = None
        for p in preds:
            if p != blank_idx and p != prev:
                collapsed.append(p)
            prev = p
        
        pred_text = tokenizer.decode_indices(collapsed)
        pred_text = unicodedata.normalize('NFC', pred_text)
        
        sample_cer = cer(pred_text, gt_text)
        sample_wer = wer(pred_text, gt_text)
        exact = pred_text == gt_text
        
        all_cer.append(sample_cer)
        all_wer.append(sample_wer)
        if exact:
            all_correct += 1
        all_total += 1
        
        if is_bangla(gt_text):
            bn_cer.append(sample_cer)
            if exact:
                bn_correct += 1
            bn_total += 1
        else:
            en_cer.append(sample_cer)
            if exact:
                en_correct += 1
            en_total += 1
        
        results.append({
            'crop_path': crop_path,
            'gt_text': gt_text,
            'pred_text': pred_text,
            'cer': sample_cer,
            'exact_match': exact
        })
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("RECOGNITION COMPLETE")
    print("="*60)
    print(f"Time: {total_time:.1f}s ({len(crops)} crops, {len(crops)/total_time:.1f}/s)")
    
    print(f"\n--- OVERALL ---")
    print(f"Words: {all_total}")
    print(f"Word Accuracy: {all_correct/max(all_total,1)*100:.2f}%")
    print(f"CER: {sum(all_cer)/max(len(all_cer),1)*100:.2f}%")
    print(f"WER: {sum(all_wer)/max(len(all_wer),1)*100:.2f}%")
    
    print(f"\n--- BANGLA ---")
    print(f"Words: {bn_total}")
    print(f"Word Accuracy: {bn_correct/max(bn_total,1)*100:.2f}%")
    print(f"CER: {sum(bn_cer)/max(len(bn_cer),1)*100:.2f}%")
    
    print(f"\n--- ENGLISH ---")
    print(f"Words: {en_total}")
    print(f"Word Accuracy: {en_correct/max(en_total,1)*100:.2f}%")
    print(f"CER: {sum(en_cer)/max(len(en_cer),1)*100:.2f}%")
    
    final_results = {
        'detection': manifest['detection_metrics'],
        'recognition': {
            'overall': {
                'num_words': all_total,
                'word_accuracy': all_correct/max(all_total,1),
                'cer': sum(all_cer)/max(len(all_cer),1),
                'wer': sum(all_wer)/max(len(all_wer),1),
            },
            'bangla': {
                'num_words': bn_total,
                'word_accuracy': bn_correct/max(bn_total,1),
                'cer': sum(bn_cer)/max(len(bn_cer),1),
            },
            'english': {
                'num_words': en_total,
                'word_accuracy': en_correct/max(en_total,1),
                'cer': sum(en_cer)/max(len(en_cer),1),
            }
        },
        'timing': {
            'detection_time': manifest['time_seconds'],
            'recognition_time': total_time,
            'total_time': manifest['time_seconds'] + total_time
        }
    }
    
    out_path = Path('stupid_again/manifests_combined/runs/last_test/e2e_crops/e2e_results.json')
    out_path.write_text(json.dumps(final_results, indent=2), encoding='utf-8')
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
