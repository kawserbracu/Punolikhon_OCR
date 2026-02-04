# Punolikhon_OCR Dataset Release

This repository does **not** store the full dataset in git.

For paper/publication, the dataset is distributed as an external archive (recommended: **Zenodo** for a DOI, or Google Drive/OneDrive).

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

## 3) Recommended distribution method (paper-ready)

### Option A: Zenodo (recommended)

1. Create a Zenodo record.
2. Upload the dataset ZIP.
3. Publish to obtain a DOI.
4. Put the DOI/link in the paper and in the main repo `README.md`.

### Option B: Google Drive / OneDrive

1. Upload the dataset ZIP.
2. Create a shareable link.
3. Put the link in the paper and in the main repo `README.md`.

## 4) Notes

- If you include very large `.bmp` files, the archive size can be very large. Consider compressing images (lossless or high-quality JPEG) if allowed by your publication policy.
- Do not include trained checkpoints in the dataset archive unless you explicitly want a combined "dataset+weights" release.
