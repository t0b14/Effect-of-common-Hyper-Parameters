from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "io"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

CONFIG_DIR = PROJECT_ROOT / "config"

MODEL_DIR = OUTPUT_DIR / "models"