import json
import unicodedata
from pathlib import Path
from typing import Dict, List


class BengaliWordOCRTokenizer:
    """
    Tokenizer for Bengali-English OCR at word level (character indices).

    Special tokens:
      <BLANK>=0 (for CTC)
      <PAD>=1   (for batching)
      <UNK>=2
    """

    BLANK = "<BLANK>"
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self) -> None:
        # Build vocabulary
        chars: List[str] = []

        # Bengali independent vowels: U+0985–U+0994
        for cp in range(0x0985, 0x0994 + 1):
            chars.append(chr(cp))

        # Bengali consonants: U+0995–U+09B9
        for cp in range(0x0995, 0x09B9 + 1):
            chars.append(chr(cp))

        # Bengali vowel signs (dependent vowels): U+09BE–U+09CC
        for cp in range(0x09BE, 0x09CC + 1):
            chars.append(chr(cp))

        # Virama/Hasant: U+09CD
        chars.append(chr(0x09CD))

        # Khanda Ta: U+09CE
        chars.append(chr(0x09CE))

        # AU length mark: U+09D7
        chars.append(chr(0x09D7))

        # Additional consonants with nukta: U+09DC–U+09DF (ড়, ঢ়, য়)
        for cp in range(0x09DC, 0x09DF + 1):
            chars.append(chr(cp))

        # Bengali diacritical marks: U+0981–U+0983 (Candrabindu, Anusvara, Visarga)
        for cp in range(0x0981, 0x0983 + 1):
            chars.append(chr(cp))

        # Bengali digits: ০–৯ (U+09E6–U+09EF)
        for cp in range(0x09E6, 0x09EF + 1):
            chars.append(chr(cp))

        # Bengali punctuation: Danda/Double Danda
        chars.append(chr(0x0964))  # ।
        chars.append(chr(0x0965))  # ॥

        # Zero Width Joiner / Non-Joiner
        chars.append('\u200D')  # ZWJ
        chars.append('\u200C')  # ZWNJ

        # English letters
        for cp in range(ord('a'), ord('z') + 1):
            chars.append(chr(cp))
        for cp in range(ord('A'), ord('Z') + 1):
            chars.append(chr(cp))

        # ASCII digits: 0–9
        for cp in range(ord('0'), ord('9') + 1):
            chars.append(chr(cp))

        # Common punctuation (extended)
        punctuation = list(".,;:!?-\"'()[]{}/<>@#$%&*+=_|\\~`")
        chars.extend(punctuation)

        # Space
        chars.append(' ')

        # Deduplicate while preserving order
        seen = set()
        dedup_chars: List[str] = []
        for ch in chars:
            if ch not in seen:
                seen.add(ch)
                dedup_chars.append(ch)

        # Special tokens first
        vocab_list = [self.BLANK, self.PAD, self.UNK] + dedup_chars
        self.char_to_idx: Dict[str, int] = {ch: i for i, ch in enumerate(vocab_list)}
        self.idx_to_char: Dict[int, str] = {i: ch for ch, i in self.char_to_idx.items()}

    def normalize_text(self, text: str) -> str:
        """
        Normalize text to NFC form for consistent Bengali representation.

        CRITICAL: All training data MUST be normalized the same way (NFC).
        Without consistent normalization, visually identical text may encode
        differently (e.g., ক্ষ as precomposed vs. sequence ক + ◌্ + ষ),
        causing mismatches.
        """
        return unicodedata.normalize('NFC', text or "")

    def encode_word(self, word: str) -> List[int]:
        word = self.normalize_text(word)
        ids: List[int] = []
        for ch in word:
            ids.append(self.char_to_idx.get(ch, self.char_to_idx[self.UNK]))
        return ids

    def decode_indices(self, indices: List[int]) -> str:
        chars: List[str] = []
        for idx in indices:
            ch = self.idx_to_char.get(int(idx), self.UNK)
            if ch in (self.BLANK, self.PAD):
                continue
            if ch == self.UNK:
                # keep a placeholder or skip; here, use '?' for unknowns
                chars.append('?')
            else:
                chars.append(ch)
        return ''.join(chars)

    def vocab_size(self) -> int:
        return len(self.char_to_idx)

    def save_vocab(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            'char_to_idx': self.char_to_idx,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str | Path) -> None:
        path = Path(path)
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        c2i = data.get('char_to_idx')
        if not isinstance(c2i, dict):
            raise ValueError('Invalid vocab file: missing char_to_idx')
        self.char_to_idx = {str(k): int(v) for k, v in c2i.items()}
        self.idx_to_char = {int(v): str(k) for k, v in self.char_to_idx.items()}


if __name__ == "__main__":
    tok = BengaliWordOCRTokenizer()
    print("Vocab size:", tok.vocab_size())

    # Test with diverse Bengali/English samples
    samples = [
        "বাংলাদেশ",      # Bangladesh
        "Bangladesh",    # English
        "ক্ষ",           # Conjunct (Ka + Hasant + Ssa)
        "য়",            # Ya with nukta
        "১২৩৪৫",        # Bengali digits
        "কলকাতা",       # Kolkata
        "প্রধানমন্ত্রী",  # Complex conjuncts
        "বাক্য।",        # Danda
        "শ্লোক॥"         # Double Danda
    ]
    for s in samples:
        enc = tok.encode_word(s)
        dec = tok.decode_indices(enc)
        match = "✓" if s == dec else "✗"
        print(f"{match} text={s} -> ids={enc[:15]}{'...' if len(enc) > 15 else ''} -> dec={dec}")

    # Save/load roundtrip
    project_root = Path(__file__).resolve().parents[1]
    tmp = project_root / "data" / "vocab.json"
    tok.save_vocab(tmp)
    tok2 = BengaliWordOCRTokenizer()
    tok2.load_vocab(tmp)
    print(f"\nLoaded vocab size: {tok2.vocab_size()}")
    print("Roundtrip test passed!" if tok.vocab_size() == tok2.vocab_size() else "Roundtrip test failed!")
