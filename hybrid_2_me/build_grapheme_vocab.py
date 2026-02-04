import argparse
import json
from pathlib import Path
from typing import Iterable, List
import sys

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_2_me.tokenizer_grapheme import BengaliGraphemeTokenizer


def load_texts(paths: Iterable[Path]) -> List[str]:
    texts: List[str] = []
    for p in paths:
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding='utf-8'))
        for it in data.get('items', []):
            texts.append(it.get('word_text') or '')
    return texts


def main():
    ap = argparse.ArgumentParser(description='Build grapheme vocab from recognition dataset')
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--output_vocab', type=str, required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    paths = [
        data_dir / 'recognition_train.json',
        data_dir / 'recognition_val.json',
        data_dir / 'recognition_test.json',
    ]

    texts = load_texts(paths)
    tok = BengaliGraphemeTokenizer()
    tok.build_vocab_from_texts(texts)
    tok.save_vocab(Path(args.output_vocab))
    print('Saved grapheme vocab to', args.output_vocab)


if __name__ == '__main__':
    main()
