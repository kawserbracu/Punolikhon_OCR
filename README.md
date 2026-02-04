# Punolikhon_OCR

Punolikhon is a two-stage bilingual (Bengali + English) handwritten OCR system.

- Stage 1 (Detection): segmentation-based word localization (Hybrid MNV3Seg).
- Stage 2 (Recognition): grapheme-level CTC recognizer (HybridHTR_STN_Transformer, `hybrid_2_me`).

This repository is prepared as a publication-ready codebase. Datasets and large binaries are not stored in git; they are provided as an external download (see `dataset/README.md`).

## Dataset Download

The page-level dataset (images + annotations) is available on Google Drive:

**Link:** https://drive.google.com/file/d/1YzaDtDB3qyBniF0wrjNi70yz-HvNa2YR/view?usp=sharing

After downloading, follow the build steps below to generate crops and manifests for training/evaluation.

## 0) Dataset: build crops + manifests (reproducible, no crops shipped)

The recognition training/evaluation in this project uses **word crops** generated from page-level annotations with 3 preprocessing modes:

- `original`
- `clahe`
- `org2`

The public dataset release should contain only **page images + annotations** (and optionally raw, see below). Users then generate crops + portable `dataset/manifests/recognition_{train,val,test}.json` locally.

Assuming your extracted dataset is located at `DATASET_ROOT`:

1) Preprocess page images into 3 modes:

```bash
python -m data.preprocess --input_dir DATASET_ROOT/images/pages --output_base DATASET_ROOT/preproc
```

2) Generate crops (d1-d6) with **relative** `crop_path` entries:

```bash
python scripts/generate_3x_crops.py \
  --preproc_base DATASET_ROOT/preproc \
  --annotations_dir DATASET_ROOT \
  --output_base DATASET_ROOT/crops_3x_combined \
  --modes original,clahe,org2 \
  --relative_to DATASET_ROOT
```

3) Optional: generate crops for `raw/` (only if you include raw pages + boxes/labels):

```bash
python scripts/generate_raw_3x.py \
  --manifest DATASET_ROOT/manifests/detection_train.json \
  --preproc_base DATASET_ROOT/preproc \
  --output_dir DATASET_ROOT/crops_3x_combined \
  --relative_to DATASET_ROOT
```

4) Split into train/val/test recognition manifests:

```bash
python scripts/merge_and_split_3x.py \
  --inputs DATASET_ROOT/crops_3x_combined/recognition_all.json DATASET_ROOT/crops_3x_combined/recognition_raw.json \
  --output_dir DATASET_ROOT/manifests
```

Then point `lassst.md` recognition paths to:

- `DATASET_ROOT/manifests/recognition_train.json`
- `DATASET_ROOT/manifests/recognition_val.json`
- `DATASET_ROOT/manifests/recognition_test.json`

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

## Raw (third-party) data note

If `raw/` comes from an external source, only redistribute it if the license/permission allows it.

- If redistribution is not allowed: exclude `raw/` from the dataset ZIP and document separate download instructions in the dataset README.
- The code supports running without raw; your published split/manifests should reflect whatever you actually release.
