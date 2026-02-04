# Punolikhon_OCR Central Paths

This file defines the canonical dataset/model/output paths used by training, evaluation, and inference scripts.

Update the values in the JSON below to match your machine. All paths may be absolute or relative to the project root.

```json
{
  "project_root": ".",

  "data_root": "dataset/manifests",
  "vocab_grapheme": "dataset/vocab/vocab_grapheme.json",

  "runs_root": "runs",

  "detection": {
    "train_json": "dataset/manifests/detection_train.json",
    "val_json": "dataset/manifests/detection_val.json",
    "test_json": "dataset/manifests/detection_test.json",
    "test_labeled_json": "dataset/manifests/detection_test_labeled_only.json",
    "model_checkpoint": "checkpoints/detection/checkpoint_best.pt",
    "mask_threshold": 0.35,
    "backbone": "hybrid"
  },

  "recognition": {
    "train_json": "dataset/manifests/recognition_train.json",
    "val_json": "dataset/manifests/recognition_val.json",
    "test_json": "dataset/manifests/recognition_test.json",
    "model_checkpoint": "checkpoints/recognition/model_best_cer.pt",
    "model_type": "hybrid_2_me",
    "decode": "greedy",
    "beam_width": 1
  },

  "e2e": {
    "images_dir": "dataset/images",
    "output_dir": "runs/e2e_outputs"
  }
}
```
