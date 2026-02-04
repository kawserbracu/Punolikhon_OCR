from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from collections import Counter


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    a1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a2 = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
    union = a1 + a2 - inter + 1e-9
    return float(inter / union)


def match_boxes(preds: List[List[float]], gts: List[List[float]], iou_threshold: float) -> List[Tuple[int, int, float]]:
    matches: List[Tuple[int, int, float]] = []
    used_p = set()
    used_g = set()
    for gi, gt in enumerate(gts):
        best_pi = -1
        best_iou = 0.0
        for pi, pr in enumerate(preds):
            if pi in used_p:
                continue
            iou = calculate_iou(pr, gt)
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_pi >= 0 and best_iou >= iou_threshold and gi not in used_g:
            used_p.add(best_pi)
            used_g.add(gi)
            matches.append((best_pi, gi, best_iou))
    return matches


def calculate_precision_recall_f1(preds: List[List[float]], gts: List[List[float]], iou_thresh: float) -> Tuple[float, float, float]:
    m = match_boxes(preds, gts, iou_thresh)
    tp = len(m)
    fp = max(0, len(preds) - tp)
    fn = max(0, len(gts) - tp)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(prec), float(rec), float(f1)


def calculate_map(predictions: List[List[List[float]]], ground_truths: List[List[List[float]]], iou_thresholds: List[float]) -> float:
    # Simple average of AP at given thresholds using precision=tp/(tp+fp), recall grid derived from matches
    aps = []
    for thr in iou_thresholds:
        precs = []
        recs = []
        for preds, gts in zip(predictions, ground_truths):
            p, r, _ = calculate_precision_recall_f1(preds, gts, thr)
            precs.append(p)
            recs.append(r)
        ap = float(np.mean(precs) * np.mean(recs))  # crude approximation
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def _levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[n, m])


def calculate_cer(predicted_text: str, ground_truth_text: str) -> float:
    if len(ground_truth_text) == 0:
        return 0.0 if len(predicted_text) == 0 else 1.0
    dist = _levenshtein(predicted_text, ground_truth_text)
    return float(dist / max(1, len(ground_truth_text)))


def calculate_wer(predicted_text: str, ground_truth_text: str) -> float:
    gt_words = ground_truth_text.split()
    pr_words = predicted_text.split()
    return calculate_cer(' '.join(pr_words), ' '.join(gt_words))
