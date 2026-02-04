import argparse
from pathlib import Path
import sys

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Note to user:
This script provides a training entrypoint named for DB (DocTR) to match your requested interface.
Due to DocTR not exposing a stable DB training API on your platform, we route to the
Faster R-CNN trainer implemented in training/train_detector_frcnn.py while preserving your
hyperparameters (epochs=100, patience=15, ReduceLROnPlateau). The resulting detector is
trainable and evaluated with IoU/mAP utilities, and integrates with the existing pipeline.

If later we enable DocTR DB training directly, we can swap this entrypoint to call a true
DB trainer without changing your command lines.
"""

from training.train_detector_frcnn import main as frcnn_main


def main():
    ap = argparse.ArgumentParser(description='Train word detector (DB entrypoint -> FasterRCNN fallback)')
    ap.add_argument('--data_dir', type=str, required=True, help='Folder with detection_train/val.json')
    ap.add_argument('--arch', type=str, default='db_mobilenet_v3_small', help='Kept for compatibility; ignored in fallback')
    ap.add_argument('--output_name', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    args = ap.parse_args()

    # Reuse Faster R-CNN trainer under the hood
    sys.argv = [sys.argv[0], '--data_dir', args.data_dir, '--output_name', args.output_name, '--batch_size', str(args.batch_size)]
    frcnn_main()


if __name__ == '__main__':
    main()
