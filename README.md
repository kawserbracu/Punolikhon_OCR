# Punolikhon_OCR

Punolikhon is a two-stage bilingual (Bengali + English) handwritten OCR system.

- Stage 1 (Detection): segmentation-based word localization (Hybrid MNV3Seg).
- Stage 2 (Recognition): grapheme-level CTC recognizer (HybridHTR_STN_Transformer, `hybrid_2_me`).

This repository is prepared as a publication-ready codebase. Datasets and large binaries are not stored in git; they are provided as an external download (see `dataset/README.md`).

## 1) Installation

```bash
pip install -r requirements.txt
```

## 2) Central configuration (`lassst.md`)

All important paths are defined in `lassst.md` (a Markdown file containing a JSON block). Scripts can read defaults from it (Option A), and you can still override any path from command-line arguments.

Before running, edit `lassst.md` so that:

- `dataset/manifests/*.json` exists (train/val/test manifests)
- `dataset/vocab/vocab_grapheme.json` exists
- `checkpoints/detection/*` and `checkpoints/recognition/*` point to your released checkpoints (or leave blank and pass CLI args)

## 3) Inference (end-to-end OCR)

```bash
python inference/ocr_pipeline.py --paths_md lassst.md
```

Common overrides:

```bash
python inference/ocr_pipeline.py \
  --paths_md lassst.md \
  --images_dir path/to/images \
  --output_dir runs/e2e_outputs
```

## 4) Evaluation

Stage-1 (detection):

```bash
python evaluation/evaluate_e2e_stage1.py --paths_md lassst.md
```

Stage-2 (recognition using saved crops/manifests):

```bash
python evaluation/evaluate_e2e_stage2.py --paths_md lassst.md
```

Recognition evaluation (grapheme tokenizer):

```bash
python evaluation/evaluate_recognition_grapheme.py --paths_md lassst.md
```

## 5) Training (optional / for reproducibility)

Detector training:

```bash
python training/train_detection_db_custom.py \
  --data_dir dataset/manifests \
  --output_name det_hybrid
```

Recognizer training (enhanced):

```bash
python training/train_hybrid_v2_enhanced.py \
  --train_json dataset/manifests/recognition_train.json \
  --val_json dataset/manifests/recognition_val.json \
  --vocab_path dataset/vocab/vocab_grapheme.json \
  --output_name rec_hybrid
```

Notes:
- Training requires GPU for practical speed.
- Inference can be executed on CPU or GPU depending on your environment.

## 6) Repository layout

- `models/`: detector/recognizer modules
- `hybrid_2_me/`: Hybrid recognizer implementation + grapheme tokenizer
- `data/`: dataset conversion and manifest utilities
- `training/`: training scripts
- `evaluation/`: evaluation scripts
- `inference/`: end-to-end pipeline
- `dataset/`: placeholder folder (download externally; see `dataset/README.md`)
- `checkpoints/`: placeholder folder for released model weights
- `runs/`: outputs (not intended to be committed)
