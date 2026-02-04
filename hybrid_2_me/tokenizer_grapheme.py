import json
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import regex as re
except Exception as exc:  # pragma: no cover - dependency required
    raise ImportError(
        "Grapheme tokenizer requires the 'regex' package (pip install regex)."
    ) from exc


class BengaliGraphemeTokenizer:
    """Grapheme-cluster tokenizer for Bengali + English OCR.

    Special tokens:
      <BLANK>=0 (for CTC)
      <PAD>=1   (for batching)
      <UNK>=2
    """

    BLANK = "<BLANK>"
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self) -> None:
        self.grapheme_to_idx: Dict[str, int] = {}
        self.idx_to_grapheme: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        self.grapheme_to_idx = {
            self.BLANK: 0,
            self.PAD: 1,
            self.UNK: 2,
        }
        self.idx_to_grapheme = {i: g for g, i in self.grapheme_to_idx.items()}

    @staticmethod
    def normalize_text(text: str) -> str:
        return unicodedata.normalize("NFC", text or "")

    @staticmethod
    def split_graphemes(text: str) -> List[str]:
        text = BengaliGraphemeTokenizer.normalize_text(text)
        return re.findall(r"\X", text)

    @staticmethod
    def _ascii_defaults() -> List[str]:
        punctuation = list(".,;:!?-\"'()[]{}/<>@#$%&*+=_|\\~`")
        letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]
        letters += [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        digits = [chr(c) for c in range(ord("0"), ord("9") + 1)]
        return letters + digits + punctuation + [" "]

    def build_vocab_from_texts(
        self,
        texts: Iterable[str],
        add_ascii: bool = True,
        add_space: bool = True,
    ) -> None:
        ordered = OrderedDict()
        for text in texts:
            for g in self.split_graphemes(text):
                ordered.setdefault(g, None)
        if add_ascii:
            for g in self._ascii_defaults():
                ordered.setdefault(g, None)
        if add_space:
            ordered.setdefault(" ", None)
        vocab_list = [self.BLANK, self.PAD, self.UNK] + list(ordered.keys())
        self.grapheme_to_idx = {g: i for i, g in enumerate(vocab_list)}
        self.idx_to_grapheme = {i: g for g, i in self.grapheme_to_idx.items()}

    def encode_word(self, word: str) -> List[int]:
        ids: List[int] = []
        for g in self.split_graphemes(word):
            ids.append(self.grapheme_to_idx.get(g, self.grapheme_to_idx[self.UNK]))
        return ids

    def decode_indices(self, indices: Sequence[int]) -> str:
        out: List[str] = []
        for idx in indices:
            g = self.idx_to_grapheme.get(int(idx), self.UNK)
            if g in (self.BLANK, self.PAD):
                continue
            if g == self.UNK:
                out.append("?")
            else:
                out.append(g)
        return "".join(out)

    def ctc_collapse(self, indices: Sequence[int], blank: int = 0) -> List[int]:
        collapsed: List[int] = []
        prev = None
        for idx in indices:
            if idx == blank:
                prev = idx
                continue
            if idx != prev:
                collapsed.append(int(idx))
            prev = idx
        return collapsed

    def decode_ctc_indices(self, indices: Sequence[int], blank: int = 0) -> str:
        return self.decode_indices(self.ctc_collapse(indices, blank=blank))

    def vocab_size(self) -> int:
        return len(self.grapheme_to_idx)

    def save_vocab(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"grapheme_to_idx": self.grapheme_to_idx}, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        g2i = data.get("grapheme_to_idx")
        if not isinstance(g2i, dict):
            raise ValueError("Invalid vocab file: missing grapheme_to_idx")
        self.grapheme_to_idx = {str(k): int(v) for k, v in g2i.items()}
        self.idx_to_grapheme = {int(v): str(k) for k, v in self.grapheme_to_idx.items()}
