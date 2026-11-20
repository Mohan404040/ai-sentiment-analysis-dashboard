"""Sentiment analysis package: data loading, preprocessing, models, prediction and ABSA."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Class order used everywhere (tables, charts, confusion matrices).
LABELS = ["Negative", "Neutral", "Positive"]
