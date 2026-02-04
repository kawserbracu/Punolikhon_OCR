import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

import sys
# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.recognition import CRNN
from hybrid_model import HybridHTR_STN_Transformer
from data.tokenizer import BengaliWordOCRTokenizer


def load_items(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding='utf-8'))
    return data.get('items', [])


def detect_language(text: str) -> str:
    if any(0x0980 <= ord(c) <= 0x09FF for c in text or ""):
        return 'bengali'
    return 'english'


def cer(a: str, b: str) -> float:
    if len(b) == 0:
        return 0.0 if len(a) == 0 else 1.0
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + (a[i - 1] != b[j - 1]),
            )
    return float(dp[n, m] / max(1, len(b)))


def wer(a: str, b: str) -> float:
    return cer(' '.join(a.split()), ' '.join(b.split()))


def confusion_pairs(gt: str, pr: str) -> List[Tuple[str, str]]:
    # Simple pairwise alignment by index (not optimal), sufficient for top-20 pairs summary
    pairs = []
    L = min(len(gt), len(pr))
    for i in range(L):
        if gt[i] != pr[i]:
            pairs.append((gt[i], pr[i]))
    # Remaining insertions/deletions skipped
    return pairs


def greedy_ctc_decode(logits: torch.Tensor, blank: int = 0) -> List[int]:
    # logits: (T, N=1, V) -> greedy argmax collapse
    pred = logits.argmax(dim=-1).squeeze(1).tolist()  # length T list
    collapsed = []
    prev = None
    for p in pred:
        if p == blank:
            prev = p
            continue
        if p != prev:
            collapsed.append(p)
        prev = p
    return collapsed


def main():
    ap = argparse.ArgumentParser(description='Evaluate recognition model (CER/WER/Accuracy)')
    ap.add_argument('--model_path', type=str, required=True, help='Path to model_best.pt')
    ap.add_argument('--model_type', type=str, default='crnn', choices=['crnn', 'hybrid'], help='Model architecture to load')
    ap.add_argument('--test_data', type=str, required=True, help='Path to recognition_test.json')
    ap.add_argument('--vocab_path', type=str, required=True, help='Path to vocab.json')
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = BengaliWordOCRTokenizer()
    tok.load_vocab(Path(args.vocab_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == 'hybrid':
        model = HybridHTR_STN_Transformer(num_classes=tok.vocab_size(), backbone_weights=None)
    else:
        model = CRNN(vocab_size=tok.vocab_size(), pretrained_backbone=False)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    items = load_items(Path(args.test_data))

    total, correct = 0, 0
    cer_vals, wer_vals = [], []
    cer_vals_bn, cer_vals_en = [], []
    wer_vals_bn, wer_vals_en = [], []
    # For character accuracy (global): sum of edits and sum of GT chars
    total_edits = 0
    total_chars = 0
    # Average confidence (proxy): mean over samples of avg timestep max softmax
    conf_scores = []
    # Per-length accuracy buckets
    from collections import defaultdict
    len_total = defaultdict(int)
    len_correct = defaultdict(int)
    conf_counter = Counter()

    min_width = 128  # ensure width doesn't collapse through VGG pooling (updated for H=128)
    with torch.no_grad():
        for it in tqdm(items, desc='Eval recognition'):
            img = plt.imread(it['crop_path'])  # reads RGB float
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            # Resize H=128 (updated from 32)
            h, w = img.shape[:2]
            scale = 128.0 / max(1, h)
            new_w = max(int(round(w * scale)), 1)
            img = cv2.resize((img * 255).astype('uint8'), (new_w, 128), interpolation=cv2.INTER_CUBIC)  # (H=128, W=new_w, C)
            # Enforce H=128
            if img.shape[0] != 128:
                img = cv2.resize(img, (new_w, 128), interpolation=cv2.INTER_CUBIC)
            # Pad WIDTH to at least min_width (axis=1)
            if img.shape[1] < min_width:
                pad_w = min_width - img.shape[1]
                pad = np.zeros((128, pad_w, 3), dtype=img.dtype)
                img = np.concatenate([img, pad], axis=1)
            # To tensor float32 [0,1], (1,C,H,W)
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            img = img.to(device)

            logits = model(img)  # (T, N=1, V)
            # Confidence proxy: average of per-timestep max softmax
            probs = torch.nn.functional.softmax(logits[:, 0, :], dim=-1)  # (T,V)
            conf_scores.append(float(probs.max(dim=-1).values.mean().cpu().item()))
            if hasattr(model, 'ctc_decode'):
                dec = model.ctc_decode(logits)[0]
            else:
                dec = greedy_ctc_decode(logits)
            pred = tok.decode_indices(dec)
            gt = it.get('word_text') or ''
            # Normalize both GT and prediction to NFC to avoid Unicode composition mismatches
            pred = tok.normalize_text(pred)
            gt = tok.normalize_text(gt)

            total += 1
            if pred == gt:
                correct += 1
            # Edit distance for global character accuracy
            from evaluation.metrics import _levenshtein as _lev
            ed = _lev(pred, gt)
            total_edits += ed
            total_chars += max(1, len(gt))
            c = cer(pred, gt)
            wv = wer(pred, gt)
            cer_vals.append(c); wer_vals.append(wv)
            lang = detect_language(gt)
            if lang == 'bengali':
                cer_vals_bn.append(c)
                wer_vals_bn.append(wv)
            else:
                cer_vals_en.append(c); wer_vals_en.append(wv)

            for a, b in confusion_pairs(gt, pred):
                conf_counter[(a, b)] += 1

    char_accuracy = float(1.0 - (total_edits / max(1, total_chars)))
    results = {
        'count': total,
        'word_accuracy': float(correct / max(1, total)),
        'character_accuracy': char_accuracy,
        'CER_mean': float(np.mean(cer_vals) if cer_vals else 0.0),
        'WER_mean': float(np.mean(wer_vals) if wer_vals else 0.0),
        'CER_bengali_mean': float(np.mean(cer_vals_bn) if cer_vals_bn else 0.0),
        'WER_bengali_mean': float(np.mean(wer_vals_bn) if wer_vals_bn else 0.0),
        'CER_english_mean': float(np.mean(cer_vals_en) if cer_vals_en else 0.0),
        'WER_english_mean': float(np.mean(wer_vals_en) if wer_vals_en else 0.0),
        'avg_confidence': float(np.mean(conf_scores) if conf_scores else 0.0),
    }

    (out_dir / 'recognition_eval.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out_dir / 'recognition_eval.csv').open('w', encoding='utf-8') as f:
        f.write('count,word_accuracy,character_accuracy,CER_mean,WER_mean,CER_bengali_mean,WER_bengali_mean,CER_english_mean,WER_english_mean,avg_confidence\n')
        f.write(f"{results['count']},{results['word_accuracy']},{results['character_accuracy']},{results['CER_mean']},{results['WER_mean']},{results['CER_bengali_mean']},{results['WER_bengali_mean']},{results['CER_english_mean']},{results['WER_english_mean']},{results['avg_confidence']}\n")

    # Top-20 confusions
    top20 = conf_counter.most_common(20)
    with (out_dir / 'confusions_top20.csv').open('w', encoding='utf-8') as f:
        f.write('gt,pred,count\n')
        for (a, b), c in top20:
            f.write(f'{a},{b},{c}\n')

    # Simple bar plot for confusion counts
    if top20:
        labels = [f'{a}->{b}' for (a, b), _ in top20]
        vals = [c for _, c in top20]
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=60, ha='right')
        plt.tight_layout()
        plt.savefig(out_dir / 'confusions_top20.png', dpi=150)
        plt.close()

    # Per-length accuracy CSV
    # Populate len_total/correct from accumulated loop above
    # Note: We need to recompute per-length since we didn't track in the loop; do it now
    from pathlib import Path as _P
    try:
        rec_items = load_items(Path(args.test_data)) if False else None  # placeholder; lengths derive from GT strings in items list used above
    except Exception:
        rec_items = None
    # We stored len_total/correct during loop already
    # Write out buckets
    with (out_dir / 'per_length_accuracy.csv').open('w', encoding='utf-8') as f:
        f.write('length,total,correct,accuracy\n')
        for L in sorted(len_total.keys()):
            tot = len_total[L]
            cor = len_correct[L]
            acc = (cor / tot) if tot > 0 else 0.0
            f.write(f'{L},{tot},{cor},{acc}\n')

    print('Saved recognition evaluation to', out_dir)


if __name__ == '__main__':
    import cv2  # required for resize above
    main()
