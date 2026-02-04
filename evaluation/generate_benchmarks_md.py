import json
from pathlib import Path
from typing import Dict

VARIANTS = {
    "original": {
        "det": Path(r"c:\Users\T2510600\Downloads\BAOCR\out\merged_all\eval_det_custom\detection_eval_custom.json"),
        "rec": Path(r"c:\Users\T2510600\Downloads\BAOCR\out\merged_all\eval_rec_finetune\recognition_eval.json"),
    },
    "clahe": {
        "det": Path(r"c:\Users\T2510600\Downloads\BAOCR\out\merged_all_clahe\eval_det_custom\detection_eval_custom.json"),
        "rec": Path(r"c:\Users\T2510600\Downloads\BAOCR\out\merged_all_clahe\eval_rec_finetune\recognition_eval.json"),
    },
    "highboost": {
        "det": Path(r"c:\Users\T2510600\Downloads\BAOCR\out\merged_all_highboost\eval_det_custom\detection_eval_custom.json"),
        "rec": Path(r"c:\Users\T2510600\Downloads\BAOCR\out\merged_all_highboost\eval_rec_finetune\recognition_eval.json"),
    },
}

DET_KEYS = ["mAP@0.5", "mAP@0.75", "Precision@0.5", "Recall@0.5", "F1@0.5"]
REC_KEYS = ["word_accuracy", "character_accuracy", "WER_mean", "CER_mean", "avg_confidence"]


def read_json(p: Path) -> Dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    rows = []
    for name, paths in VARIANTS.items():
        det = read_json(paths["det"])
        rec = read_json(paths["rec"])
        row = {
            "variant": name,
            **{k: det.get(k, None) for k in DET_KEYS},
            **{k: rec.get(k, None) for k in REC_KEYS},
        }
        rows.append(row)

    # Build markdown
    headers = [
        "Variant",
        "mAP@0.5",
        "mAP@0.75",
        "Precision@0.5",
        "Recall@0.5",
        "F1@0.5",
        "Word Acc",
        "Char Acc",
        "WER",
        "CER",
        "Avg Conf",
    ]
    def fmt(x):
        if isinstance(x, (int, float)):
            return f"{x:.4f}"
        return "-" if x is None else str(x)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        line = [
            r["variant"],
            fmt(r.get("mAP@0.5")),
            fmt(r.get("mAP@0.75")),
            fmt(r.get("Precision@0.5")),
            fmt(r.get("Recall@0.5")),
            fmt(r.get("F1@0.5")),
            fmt(r.get("word_accuracy")),
            fmt(r.get("character_accuracy")),
            fmt(r.get("WER_mean")),
            fmt(r.get("CER_mean")),
            fmt(r.get("avg_confidence")),
        ]
        lines.append("| " + " | ".join(line) + " |")

    out_dir = Path(r"c:\Users\T2510600\Downloads\BAOCR\out\benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "benchmarks.md"
    content = "\n".join([
        "# BAOCR Benchmarks (Detection + Recognition)",
        "",
        *lines,
        "",
        "Notes:",
        "- Detection metrics at IoU=0.5 and 0.75 are from custom MNV3 segmentation evaluator.",
        "- Recognition metrics are computed on the test crops for each preprocessing variant.",
    ])
    out_md.write_text(content, encoding="utf-8")
    print("Wrote:", out_md)


if __name__ == "__main__":
    main()
