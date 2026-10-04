"""Download and verify models without opening a webcam; safe to run again offline."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_tracker import ensure_model
from scene import ensure_person_model
from segmentation import MODEL_DIR, ensure_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-only", action="store_true",
                        help="prepare only EdgeSAM, for real-model fixture tests")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    ensure_models(args.model_dir)
    if not args.segmentation_only:
        ensure_model()
        ensure_person_model()
    print("Models verified. Ready for offline use.")


if __name__ == "__main__":
    main()
