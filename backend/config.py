import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
E4E_REPO = (PROJECT_ROOT.parent / "encoder4editing").resolve()

E4E_CKPT = PROJECT_ROOT / "runs" / "e4e2" / "checkpoints" / "best_model.pt"
PREDICTOR = PROJECT_ROOT / "pretrained" / "shape_predictor_68_face_landmarks.dat"
DIRECTIONS_DIR = PROJECT_ROOT / "directions"

STORAGE = Path(__file__).resolve().parent / "storage"

DEVICE = os.environ.get("DEVICE", "cuda")
