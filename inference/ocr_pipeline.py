from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import sys

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detection import build_detection_model
from models.detection_mnv3seg import MNV3Seg
from models.recognition import CRNN
from data.tokenizer import BengaliWordOCRTokenizer
from baocr_paths import load_paths_from_md
from hybrid_2_me import HybridHTR_STN_Transformer
from hybrid_2_me.tokenizer_grapheme import BengaliGraphemeTokenizer
from evaluation.evaluate_recognition_grapheme import greedy_ctc_decode


class EndToEndOCR:
    def __init__(
        self,
        detection_model_path: str | None,
        recognition_model_path: str,
        tokenizer_path: str,
        custom_det_model: str | None = None,
        custom_backbone: str = 'large',
        mask_threshold: float = 0.3,
        recognition_model_type: str = 'crnn',
    ):
        # Detection: DocTR predictor by default, or custom MNV3Seg if provided
        self.use_custom_det = custom_det_model is not None
        self.mask_threshold = float(mask_threshold)
        self.recognition_model_type = str(recognition_model_type)
        if self.use_custom_det:
            state = torch.load(custom_det_model, map_location='cpu')
            bb = state.get('backbone', custom_backbone)
            self.custom = MNV3Seg(backbone=bb, pretrained=False)
            self.custom.load_state_dict(state['state_dict'])
            self.custom.eval()
        else:
            self.detector = build_detection_model(pretrained=True)
        if self.recognition_model_type == 'hybrid_2_me':
            self.tok = BengaliGraphemeTokenizer()
            self.tok.load_vocab(Path(tokenizer_path))
            self.blank_idx = self.tok.grapheme_to_idx.get(self.tok.BLANK, 0)
            self.recognizer = HybridHTR_STN_Transformer(num_classes=self.tok.vocab_size())
            if recognition_model_path:
                state = torch.load(recognition_model_path, map_location='cpu')
                if isinstance(state, dict) and 'state_dict' in state:
                    self.recognizer.load_state_dict(state['state_dict'])
                else:
                    self.recognizer.load_state_dict(state)
            self.recognizer.eval()
        else:
            self.tok = BengaliWordOCRTokenizer()
            self.tok.load_vocab(Path(tokenizer_path))
            self.recognizer = CRNN(vocab_size=self.tok.vocab_size(), pretrained_backbone=False)
            if recognition_model_path:
                state = torch.load(recognition_model_path, map_location='cpu')
                self.recognizer.load_state_dict(state)
            self.recognizer.eval()

    def detect_words(self, image: np.ndarray) -> List[List[float]]:
        # returns normalized boxes [x1,y1,x2,y2]
        if self.use_custom_det:
            return self._detect_words_custom(image)
        else:
            from doctr.io.image import read_img_as_tensor
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tmp = Path("_tmp_det.jpg")
            cv2.imwrite(str(tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            page = read_img_as_tensor(str(tmp))
            preds = self.detector([page])
            tmp.unlink(missing_ok=True)
            boxes: List[List[float]] = []
            # Handle different predictor return types across doctr versions
            obj = preds[0] if isinstance(preds, (list, tuple)) and len(preds) > 0 else preds
            if hasattr(obj, 'pages'):
                for p in obj.pages:
                    for l in getattr(p, 'lines', []):
                        for w in getattr(l, 'words', []):
                            poly = np.array(w.geometry, dtype=float)
                            xs = poly[:, 0]; ys = poly[:, 1]
                            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
            elif isinstance(obj, dict):
                pages = obj.get('pages', [])
                for p in pages:
                    for blk in p.get('blocks', []):
                        for ln in blk.get('lines', []):
                            for wd in ln.get('words', []):
                                poly = np.array(wd.get('geometry', []), dtype=float)
                                if poly.size == 0:
                                    continue
                                xs = poly[:, 0]; ys = poly[:, 1]
                                boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
            return boxes

    def _detect_words_custom(self, image: np.ndarray) -> List[List[float]]:
        # Preprocess to 768x1024, forward through MNV3Seg, threshold to mask, contours->boxes
        H, W = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (1024, 768), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            logits = self.custom(img_t)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        # Resize back to original size
        mask = cv2.resize(probs, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask >= self.mask_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[List[float]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 5 or h < 5:
                continue
            x1 = x / W; y1 = y / H; x2 = (x + w) / W; y2 = (y + h) / H
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
        return boxes

    @torch.no_grad()
    def recognize_word(self, crop_image: np.ndarray) -> str:
        # Prepare crop (H=32)
        rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        if self.recognition_model_type == 'hybrid_2_me':
            h, w = rgb.shape[:2]
            scale = 128.0 / max(1, h)
            new_w = max(int(round(w * scale)), 1)
            rgb = cv2.resize(rgb, (new_w, 128), interpolation=cv2.INTER_CUBIC)
            if rgb.shape[1] < 32:
                rgb = cv2.copyMakeBorder(rgb, 0, 0, 0, 32 - rgb.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            logits = self.recognizer(x)
            ids = greedy_ctc_decode(logits, blank=self.blank_idx)
            return self.tok.decode_indices(ids)
        h, w = rgb.shape[:2]
        scale = 32.0 / max(1, h)
        new_w = max(int(round(w * scale)), 1)
        rgb = cv2.resize(rgb, (new_w, 32), interpolation=cv2.INTER_AREA)
        min_width = 32
        if rgb.shape[1] < min_width:
            pad_w = min_width - rgb.shape[1]
            pad = np.zeros((32, pad_w, 3), dtype=rgb.dtype)
            rgb = np.concatenate([rgb, pad], axis=1)
        img = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        logits = self.recognizer(img)
        ids = self.recognizer.ctc_decode(logits)[0]
        return self.tok.decode_indices(ids)

    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        img = cv2.imread(image_path)
        if img is None:
            return []
        H, W = img.shape[:2]
        boxes = self.detect_words(img)
        results: List[Dict[str, Any]] = []
        for b in boxes:
            x1 = int(round(b[0] * W)); y1 = int(round(b[1] * H))
            x2 = int(round(b[2] * W)); y2 = int(round(b[3] * H))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            text = self.recognize_word(crop)
            results.append({'box': b, 'text': text})
        return results

    def visualize(self, image: np.ndarray, results: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]] | None = None) -> np.ndarray:
        out = image.copy()
        H, W = out.shape[:2]
        for r in results:
            x1 = int(round(r['box'][0] * W)); y1 = int(round(r['box'][1] * H))
            x2 = int(round(r['box'][2] * W)); y2 = int(round(r['box'][3] * H))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, r['text'], (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return out


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description='Run end-to-end OCR on a folder of images and save visualizations/results')
    ap.add_argument('--paths_md', type=str, default=str(PROJECT_ROOT / 'lassst.md'))
    ap.add_argument('--images_dir', type=str, default='')
    ap.add_argument('--rec_model', type=str, default='', help='Path to trained recognition model_best.pt')
    ap.add_argument('--vocab', type=str, default='', help='Path to vocab.json / vocab_grapheme.json')
    ap.add_argument('--output_dir', type=str, default='')
    ap.add_argument('--limit', type=int, default=0, help='Optional limit on number of images')
    ap.add_argument('--custom_det_model', type=str, default=None, help='Path to custom MNV3Seg model_best.pt to use instead of DocTR predictor')
    ap.add_argument('--custom_backbone', type=str, default='large', choices=['large','small'])
    ap.add_argument('--mask_threshold', type=float, default=0.3, help='Threshold for custom detector mask->boxes')
    ap.add_argument('--rec_model_type', type=str, default='', help='crnn or hybrid_2_me (defaults from lassst.md if empty)')
    args = ap.parse_args()

    paths = load_paths_from_md(args.paths_md) if args.paths_md else None
    images_dir = args.images_dir or (str(Path(paths.get_nested('e2e', 'images_dir', default=''))) if paths else '')
    output_dir = args.output_dir or (str(Path(paths.get_nested('e2e', 'output_dir', default=str(PROJECT_ROOT / 'runs' / 'ocr_vis')))) ) if paths else str(PROJECT_ROOT / 'runs' / 'ocr_vis')
    rec_model = args.rec_model or (str(Path(paths.get_nested('recognition', 'model_checkpoint', default=''))) if paths else '')
    vocab = args.vocab or (str(Path(paths.get('vocab_grapheme', ''))) if paths else '')
    rec_model_type = args.rec_model_type or (str(paths.get_nested('recognition', 'model_type', default='crnn')) if paths else 'crnn')

    # Default custom detector checkpoint/threshold from lassst.md (Option A: CLI overrides win)
    custom_det_model = args.custom_det_model
    if custom_det_model is None and paths is not None:
        custom_det_model = paths.get_nested('detection', 'model_checkpoint', default=None)
    mask_threshold = float(args.mask_threshold)
    if (args.mask_threshold == 0.3) and paths is not None:
        mask_threshold = float(paths.get_nested('detection', 'mask_threshold', default=mask_threshold))

    if not images_dir:
        raise ValueError('images_dir is required (pass --images_dir or set e2e.images_dir in lassst.md)')
    if not output_dir:
        raise ValueError('output_dir is required (pass --output_dir or set e2e.output_dir in lassst.md)')
    if not rec_model:
        raise ValueError('rec_model is required (pass --rec_model or set recognition.model_checkpoint in lassst.md)')
    if not vocab:
        raise ValueError('vocab is required (pass --vocab or set vocab_grapheme in lassst.md)')

    images_dir = Path(images_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = EndToEndOCR(
        detection_model_path=None,
        recognition_model_path=rec_model,
        tokenizer_path=vocab,
        custom_det_model=custom_det_model,
        custom_backbone=args.custom_backbone,
        mask_threshold=mask_threshold,
        recognition_model_type=rec_model_type,
    )

    # Collect images
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    imgs = [p for p in images_dir.rglob('*') if p.suffix.lower() in exts]
    if args.limit and args.limit > 0:
        imgs = imgs[: args.limit]

    all_results: Dict[str, Any] = {}
    for p in imgs:
        res = ocr.process_image(str(p))
        all_results[str(p)] = res
        # Save visualization
        img = cv2.imread(str(p))
        if img is not None:
            vis = ocr.visualize(img, res)
            cv2.imwrite(str(out_dir / f"{p.stem}_vis{p.suffix}"), vis)

    # Save results JSON
    (out_dir / 'results.json').write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved OCR results and visualizations to', out_dir)


if __name__ == '__main__':
    main()
