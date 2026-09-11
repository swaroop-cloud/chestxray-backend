"""
Loads the trained Keras model once and reuses it across requests.
Point MODEL_PATH at whichever .keras file you want to serve
(e.g. weighted_cnn_chest_xray.keras).
"""

import os
from pathlib import Path

from tensorflow import keras  # pyright: ignore

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).parent / "models" / "weighted_cnn_chest_xray.keras"),
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Copy your .keras file into backend/models/ or set the "
                "MODEL_PATH environment variable."
            )
        _model = keras.models.load_model(MODEL_PATH, compile=False)
    return _model
