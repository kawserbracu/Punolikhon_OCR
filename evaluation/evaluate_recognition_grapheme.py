import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import sys
import numpy as np
import torch
from tqdm import tqdm
import cv2
import unicodedata

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.recognition import CRNN
from hybrid_2_me import HybridHTR_STN_Transformer, HybridHTR_STN_Transformer_BiLSTM
from hybrid_2_me.tokenizer_grapheme import BengaliGraphemeTokenizer
from baocr_paths import load_paths_from_md
import re

# Bangla Unicode ranges for vowel signs (matras)
BANGLA_VOWEL_SIGNS = set([
    '\u09BE',  # া (AA)
    '\u09BF',  # ি (I)
    '\u09C0',  # ী (II)
    '\u09C1',  # ু (U)
    '\u09C2',  # ূ (UU)
    '\u09C3',  # ৃ (R)
    '\u09C4',  # ৄ (RR)
    '\u09C7',  # ে (E)
    '\u09C8',  # ৈ (AI)
    '\u09CB',  # ো (O)
    '\u09CC',  # ৌ (AU)
    '\u09CD',  # ্ (Virama/Hasanta)
])

# Bangla consonants range: \u0995-\u09B9
BANGLA_CONSONANT_RANGE = (0x0995, 0x09B9)


def is_bangla_char(c: str) -> bool:
    """Check if character is in Bangla Unicode block (U+0980 to U+09FF)."""
    return '\u0980' <= c <= '\u09FF'


def is_ascii(c: str) -> bool:
    """Check if character is ASCII (English)."""
    return ord(c) < 128


def bangla_postprocess(text: str) -> str:
    """
    Minimal Bangla-aware CTC post-processing.
    
    Only removes consecutive duplicate vowel signs (matras) caused by CTC repetition.
    This is the safest fix that won't remove valid characters.
    """
    if not text:
        return text
    
    result = []
    for c in text:
        # Only skip if this is a Bangla vowel sign AND it's a duplicate of the previous char
        if c in BANGLA_VOWEL_SIGNS and result and result[-1] == c:
            continue
        result.append(c)
    
    return ''.join(result)


def _char_diff(gt: str, pred: str) -> str:
    """Generate a simple character-level diff for error analysis."""
    diff_parts = []
    max_len = max(len(gt), len(pred))
    for i in range(max_len):
        g = gt[i] if i < len(gt) else ''
        p = pred[i] if i < len(pred) else ''
        if g == p:
            diff_parts.append(g)
        elif g and p:
            diff_parts.append(f'[{g}→{p}]')
        elif g:
            diff_parts.append(f'[-{g}]')
        else:
            diff_parts.append(f'[+{p}]')
    return ''.join(diff_parts)


def load_items(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding='utf-8'))
    return data.get('items', [])


def greedy_ctc_decode(logits: torch.Tensor, blank: int = 0) -> List[int]:
    pred = logits.argmax(dim=-1).squeeze(1).tolist()
    collapsed: List[int] = []
    prev = None
    for p in pred:
        if p == blank:
            prev = p
            continue
        if p != prev:
            collapsed.append(p)
        prev = p
    return collapsed


def _log_add(a: float, b: float) -> float:
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a < b:
        a, b = b, a
    return a + float(torch.log1p(torch.exp(torch.tensor(b - a))))


def ctc_beam_search(logits: torch.Tensor, beam_width: int = 10, blank: int = 0) -> List[int]:
    """Prefix beam search for CTC. Returns best path (list of ids)."""
    log_probs = torch.log_softmax(logits, dim=-1).squeeze(1)  # (T,V)
    beams = {(): (0.0, float('-inf'))}  # prefix -> (p_blank, p_non_blank)

    for t in range(log_probs.size(0)):
        new_beams = {}
        step = log_probs[t]
        for prefix, (p_b, p_nb) in beams.items():
            for c in range(step.size(0)):
                p = float(step[c])
                if c == blank:
                    nb = new_beams.get(prefix, (float('-inf'), float('-inf')))
                    new_p_b = _log_add(nb[0], _log_add(p_b + p, p_nb + p))
                    new_beams[prefix] = (new_p_b, nb[1])
                    continue

                last = prefix[-1] if prefix else None
                new_prefix = prefix + (c,)
                nb = new_beams.get(new_prefix, (float('-inf'), float('-inf')))

                if c == last:
                    new_p_nb = _log_add(nb[1], p_b + p)
                else:
                    new_p_nb = _log_add(nb[1], _log_add(p_b + p, p_nb + p))

                new_beams[new_prefix] = (nb[0], new_p_nb)

        # Prune to top beam_width
        beams = {}
        scored = []
        for prefix, (p_b, p_nb) in new_beams.items():
            score = _log_add(p_b, p_nb)
            scored.append((score, prefix, p_b, p_nb))
        scored.sort(key=lambda x: x[0], reverse=True)
        for score, prefix, p_b, p_nb in scored[:beam_width]:
            beams[prefix] = (p_b, p_nb)

    best_prefix = max(beams.items(), key=lambda x: _log_add(x[1][0], x[1][1]))[0]
    return list(best_prefix)


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


def char_accuracy(pred: str, gt: str) -> float:
    """Calculate character-level accuracy."""
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    correct = sum(1 for p, g in zip(pred, gt) if p == g)
    return correct / len(gt)


def is_bengali_text(text: str) -> bool:
    """Check if text contains primarily Bengali characters."""
    if not text:
        return False
    bengali_count = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    return bengali_count > len(text) * 0.5


def is_english_text(text: str) -> bool:
    """Check if text contains primarily English/ASCII characters."""
    if not text:
        return False
    ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    return ascii_alpha > len(text) * 0.5


def main():
    ap = argparse.ArgumentParser(description='Evaluate recognition with grapheme tokenizer')
    ap.add_argument('--paths_md', type=str, default=str(PROJECT_ROOT / 'lassst.md'))
    ap.add_argument('--model_path', type=str, default='')
    ap.add_argument('--model_type', type=str, default='', choices=['crnn', 'hybrid_2_me', 'hybrid_2_me_bilstm'])
    ap.add_argument('--test_data', type=str, default='')
    ap.add_argument('--vocab_path', type=str, default='')
    ap.add_argument('--output_dir', type=str, default='')
    ap.add_argument('--max_samples', type=int, default=0)
    ap.add_argument('--decode', type=str, default='', choices=['beam', 'greedy'])
    ap.add_argument('--beam_width', type=int, default=0)
    ap.add_argument('--save_every', type=int, default=200, help='Save partial results every N samples')
    ap.add_argument('--dump_errors', type=int, default=0, help='Dump N random error samples for inspection (0=disabled)')
    args = ap.parse_args()

    # Option A: defaults come from lassst.md, but explicit CLI values override
    paths = load_paths_from_md(args.paths_md) if args.paths_md else None
    if paths is not None:
        if not args.model_path:
            args.model_path = str(Path(paths.get_nested('recognition', 'model_checkpoint', default='')))
        if not args.test_data:
            args.test_data = str(Path(paths.get_nested('recognition', 'test_json', default='')))
        if not args.vocab_path:
            args.vocab_path = str(Path(paths.get('vocab_grapheme', '')))
        if not args.output_dir:
            runs_root = paths.get('runs_root', '')
            out_default = Path(runs_root) / 'eval_recognition' if runs_root else (PROJECT_ROOT / 'runs' / 'eval_recognition')
            args.output_dir = str(out_default)
        if not args.model_type:
            args.model_type = str(paths.get_nested('recognition', 'model_type', default='hybrid_2_me'))
        if not args.decode:
            args.decode = str(paths.get_nested('recognition', 'decode', default='greedy'))
        if not args.beam_width or int(args.beam_width) <= 0:
            args.beam_width = int(paths.get_nested('recognition', 'beam_width', default=10))

    if not args.model_path:
        raise ValueError('model_path is required (pass --model_path or set recognition.model_checkpoint in lassst.md)')
    if not args.test_data:
        raise ValueError('test_data is required (pass --test_data or set recognition.test_json in lassst.md)')
    if not args.vocab_path:
        raise ValueError('vocab_path is required (pass --vocab_path or set vocab_grapheme in lassst.md)')
    if not args.output_dir:
        raise ValueError('output_dir is required (pass --output_dir or set runs_root in lassst.md)')
    if not args.model_type:
        args.model_type = 'hybrid_2_me'
    if not args.decode:
        args.decode = 'greedy'
    if not args.beam_width or int(args.beam_width) <= 0:
        args.beam_width = 10

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = BengaliGraphemeTokenizer()
    tok.load_vocab(Path(args.vocab_path))
    blank_idx = tok.grapheme_to_idx.get(tok.BLANK, 0)

    if args.decode == 'greedy' and args.beam_width != 1:
        print('Warning: greedy decoding ignores beam_width; forcing beam_width=1')
        args.beam_width = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == 'crnn':
        model = CRNN(vocab_size=tok.vocab_size(), pretrained_backbone=False)
    elif args.model_type == 'hybrid_2_me_bilstm':
        model = HybridHTR_STN_Transformer_BiLSTM(num_classes=tok.vocab_size())
    else:
        model = HybridHTR_STN_Transformer(num_classes=tok.vocab_size())
    state = torch.load(args.model_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    items = load_items(Path(args.test_data))
    if args.max_samples and args.max_samples > 0:
        items = items[: args.max_samples]

    total_cer = 0.0
    total_wer = 0.0
    correct_words = 0
    total = 0
    total_char_acc = 0.0
    total_confidence = 0.0
    # Language-specific metrics
    bengali_cer = 0.0
    bengali_wer = 0.0
    bengali_count = 0
    bengali_correct_words = 0
    english_cer = 0.0
    english_wer = 0.0
    english_count = 0
    english_correct_words = 0
    error_samples = []  # For error dump

    partial_json = out_dir / 'recognition_eval_grapheme_partial.json'
    partial_csv = out_dir / 'recognition_eval_grapheme_partial.csv'
    if not partial_csv.exists():
        partial_csv.write_text('count,word_accuracy,CER_mean,WER_mean,decode,beam_width\n', encoding='utf-8')

    def save_partial() -> None:
        if total <= 0:
            return
        results = {
            'count': total,
            'word_accuracy': float(correct_words / max(1, total)),
            'CER_mean': float(total_cer / max(1, total)),
            'WER_mean': float(total_wer / max(1, total)),
            'model_type': args.model_type,
            'decode': args.decode,
            'beam_width': int(args.beam_width),
        }
        partial_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        with partial_csv.open('a', encoding='utf-8') as f:
            f.write(f"{results['count']},{results['word_accuracy']},{results['CER_mean']},{results['WER_mean']},{args.decode},{args.beam_width}\n")

    try:
        for it in tqdm(items, desc='Eval'):
            img = cv2.imread(it['crop_path'])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            scale = 128.0 / max(1, h)
            new_w = max(int(round(w * scale)), 1)
            img = cv2.resize(img, (new_w, 128), interpolation=cv2.INTER_CUBIC)
            if img.shape[1] < 32:
                pad_w = 32 - img.shape[1]
                img = cv2.copyMakeBorder(img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            x = x.to(device)
            with torch.no_grad():
                logits = model(x)
            if args.decode == 'greedy':
                pred_ids = greedy_ctc_decode(logits, blank=blank_idx)
            else:
                pred_ids = ctc_beam_search(logits, beam_width=args.beam_width, blank=blank_idx)
            pred_text = tok.decode_indices(pred_ids)
            # pred_text = bangla_postprocess(pred_text)  # Disabled - no benefit observed
            gt_text = it.get('word_text') or ''
            # Apply Unicode NFC normalization to fix grapheme cluster comparison issues
            # This ensures য় (U+09AF U+09BC) is compared correctly regardless of encoding
            pred_text = unicodedata.normalize('NFC', pred_text)
            gt_text = unicodedata.normalize('NFC', gt_text)
            sample_cer = cer(pred_text, gt_text)
            sample_wer = wer(pred_text, gt_text)
            sample_char_acc = char_accuracy(pred_text, gt_text)
            total_cer += sample_cer
            total_wer += sample_wer
            total_char_acc += sample_char_acc
            
            # Calculate confidence from logits
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1).values
                sample_conf = float(max_probs.mean())
            total_confidence += sample_conf
            
            # Language-specific metrics
            is_bn = is_bengali_text(gt_text)
            is_en = (not is_bn) and is_english_text(gt_text)
            if is_bn:
                bengali_cer += sample_cer
                bengali_wer += sample_wer
                bengali_count += 1
            elif is_en:
                english_cer += sample_cer
                english_wer += sample_wer
                english_count += 1
            
            is_correct = (pred_text == gt_text)
            if is_correct:
                correct_words += 1
                if is_bn:
                    bengali_correct_words += 1
                elif is_en:
                    english_correct_words += 1
            else:
                # Collect error samples for dump
                if args.dump_errors > 0 and len(error_samples) < args.dump_errors * 3:
                    error_samples.append({
                        'crop_path': it.get('crop_path', ''),
                        'gt': gt_text,
                        'pred': pred_text,
                        'cer': sample_cer,
                        'diff': _char_diff(gt_text, pred_text)
                    })
            total += 1
            if args.save_every > 0 and total % args.save_every == 0:
                save_partial()
    except KeyboardInterrupt:
        save_partial()
        print('Interrupted: partial results saved to', partial_json)
        return

    results = {
        'count': total,
        'word_accuracy': float(correct_words / max(1, total)),
        'bengali_word_accuracy': float(bengali_correct_words / max(1, bengali_count)) if bengali_count > 0 else None,
        'english_word_accuracy': float(english_correct_words / max(1, english_count)) if english_count > 0 else None,
        'character_accuracy': float(total_char_acc / max(1, total)),
        'CER_mean': float(total_cer / max(1, total)),
        'WER_mean': float(total_wer / max(1, total)),
        'CER_bengali_mean': float(bengali_cer / max(1, bengali_count)) if bengali_count > 0 else None,
        'WER_bengali_mean': float(bengali_wer / max(1, bengali_count)) if bengali_count > 0 else None,
        'CER_english_mean': float(english_cer / max(1, english_count)) if english_count > 0 else None,
        'WER_english_mean': float(english_wer / max(1, english_count)) if english_count > 0 else None,
        'avg_confidence': float(total_confidence / max(1, total)),
        'bengali_count': bengali_count,
        'english_count': english_count,
        'model_type': args.model_type,
        'decode': args.decode,
        'beam_width': int(args.beam_width),
    }

    out_json = out_dir / 'recognition_eval_grapheme.json'
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved recognition eval to', out_json)

    # Save error dump if requested
    if args.dump_errors > 0 and error_samples:
        import random
        random.shuffle(error_samples)
        dump_samples = error_samples[:args.dump_errors]
        dump_json = out_dir / 'error_dump.json'
        dump_json.write_text(json.dumps(dump_samples, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Saved {len(dump_samples)} error samples to', dump_json)


if __name__ == '__main__':
    main()
