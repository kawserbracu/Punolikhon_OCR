# Punolikhon_OCR Dataset Release

This repository does **not** store the full dataset in git.

For paper/publication, the dataset is distributed as an external archive (recommended: **Zenodo** for a DOI, or Google Drive/OneDrive).

## Download Link (Google Drive)

**Link:** https://drive.google.com/file/d/1YzaDtDB3qyBniF0wrjNi70yz-HvNa2YR/view?usp=sharing

This release is intended to be **page-level** (images + annotations). Word-crops and crop-based manifests are generated locally using the scripts in this repository.

## 1) What the dataset contains

Based on the project setup, the dataset release is expected to include:

- Page images from multiple sources: `d1/`, `d2/`, `d3/`, `d4/`, `d5/`, `d6/`, and `raw/`
- Annotations:
  - `d1_annotations.json`, `d2_annotations.json`, `d3_annotations.json`, `d4_annotations.json`
  - `d5_annotations/` (per-image JSON)
  - `d6_annotations/` (per-image JSON)
  - optional Label Studio exports (e.g., `d5_annotations_ls.json`, `d6_annotations_ls.json`)
- Pre-built manifests for training/evaluation:
  - `dataset/manifests/detection_{train,val,test}.json`
  - `dataset/manifests/recognition_{train,val,test}.json`
  - optional: `detection_test_labeled_only.json`
- Grapheme vocabulary:
  - `dataset/vocab/vocab_grapheme.json`

## 2) Expected folder structure after download

After extracting the dataset archive, place the contents so that the repository has:

- `dataset/images/` (or multiple subfolders; update `lassst.md` accordingly)
- `dataset/manifests/`
- `dataset/vocab/`

Then update `lassst.md` so it points to the extracted paths.

## 3) Build crops + recognition manifests (no crops shipped)

The recognition model trains/evaluates on **word crops** generated in 3 preprocessing modes:

- `original`
- `clahe`
- `org2`

Assuming the extracted dataset root is `DATASET_ROOT`:

1) Preprocess page images into 3 modes:

```bash
python -m data.preprocess --input_dir DATASET_ROOT/images/pages --output_base DATASET_ROOT/preproc
```

2) Generate crops (d1-d6) with relative `crop_path`:

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

After this, set `lassst.md` recognition paths to `DATASET_ROOT/manifests/recognition_{train,val,test}.json`.

## 4) Recommended distribution method (paper-ready)

### Option A: Zenodo (recommended)

1. Create a Zenodo record.
2. Upload the dataset ZIP.
3. Publish to obtain a DOI.
4. Put the DOI/link in the paper and in the main repo `README.md`.

### Option B: Google Drive / OneDrive

1. Upload the dataset ZIP.
2. Create a shareable link.
3. Put the link in the paper and in the main repo `README.md`.

## 5) Notes

- If you include very large `.bmp` files, the archive size can be very large. Consider compressing images (lossless or high-quality JPEG) if allowed by your publication policy.
- Do not include trained checkpoints in the dataset archive unless you explicitly want a combined "dataset+weights" release.

### Raw (third-party) data

If `raw/` originates from an external dataset, only redistribute it if the original license/permission allows redistribution.

- If redistribution is not allowed: exclude `raw/` from the archive and provide separate download instructions.
- If redistribution is allowed: include `images/pages/raw/` and any required annotations/manifests so the optional raw crop-generation step works.
