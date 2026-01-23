import logging
from pathlib import Path
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(os.environ.get("DATA_DIR", "."))  # default: current directory
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))  # default: models in current directory
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))  # default: current directory

def _load_model(name):
    """Load a Keras model."""
    path = MODELS_DIR / name
    try:
        m = keras.models.load_model(str(path))
        logger.info("Loaded model: %s", name)
        return m
    except Exception as exc:
        logger.warning("Could not load model %s: %s", name, exc)
        return None

def read_to_test_data(interp_path):
    """Read and normalize FRB test images."""
    with np.load(interp_path) as frbdata:
        frbd1 = frbdata["arr_0"]

    frbdn = []
    for img in frbd1:
        immax = img.max()
        frbdn.append(img / (immax / 255.0))

    frbdn = np.asarray(frbdn)
    print(frbdn.shape)
    frbdn.shape += 1,
    logger.info("Loaded %d images, shape=%s", frbdn.shape[0], frbdn.shape)
    return frbdn

def conf_matrix_from_pred(predictions, n):
    """Convert binary-model predictions into a numeric 5x5 matrix."""
    def _get_score(pred_array, idx):
        arr = np.asarray(pred_array)
        if arr.ndim == 1:
            val = arr[idx]
        else:
            val = arr[idx].ravel()[0]
        return float(val)

    map_indices = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # A vs B,C,C1,C2
        (1, 2), (1, 3), (1, 4),          # B vs C,C1,C2
        (2, 3), (2, 4),                  # C vs C1,C2
        (3, 4)                           # C1 vs C2
    ]

    mat = np.zeros((5, 5), dtype=np.float32)
    mat[mat==0]=2

    for pred_idx, (r, c) in enumerate(map_indices):
        score = _get_score(predictions[pred_idx], n)
        mat[r, c] = round(score, 4)
        mat[c, r] = round(1.0 - score, 4)
    mat[mat==2] = np.nan
    
    return mat.T

# Load models
model_names = [
    "model_IvsII", "model_IvsIII", "model_IvsIV", "model_IvsV",
    "model_IIvsIII", "model_IIvsIV", "model_IIvsV",
    "model_IIIvsIV", "model_IIIvsV", "model_IVvsV"
]

models = [ _load_model(name) for name in model_names]




# Load test data
interp_path = DATA_DIR / "chime_interp_frbs.npz"
if not interp_path.exists():
    logger.error("Data file %s is missing.", interp_path.as_posix())
    exit(1)

test_images = read_to_test_data(interp_path)


logger.info("Running predictions on 2-D dynamic spectra...")
# Generate predictions
predictions = []

for model in models:
    predictions.append(model(test_images)) 


logger.info("Predictions on 2-D dynamic spectra completed.")
#logger.info("predictions shape: %s", predictions.shape)
print(len(predictions))
print(len(predictions[0]))

# Thresholds matrix
thresholds = np.zeros((5, 5))
thresholds[0][1] = 0.64    # A vs B
thresholds[0][2] = 0.5     # A vs C
thresholds[0][3] = 0.5     # A vs C1
thresholds[0][4] = 0.5     # A vs C2
thresholds[1][0] = 0.36    # B vs A
thresholds[1][2] = 0.5     # B vs C
thresholds[1][3] = 0.2     # B vs C1
thresholds[1][4] = 0.5     # B vs C2
thresholds[2][0] = 0.5     # C vs A
thresholds[2][1] = 0.5     # C vs B
thresholds[2][3] = 0.5     # C vs C1
thresholds[2][4] = 0.5     # C vs C2
thresholds[3][0] = 0.5     # C1 vs A
thresholds[3][1] = 0.8     # C1 vs B
thresholds[3][2] = 0.5     # C1 vs C
thresholds[3][4] = 0.5     # C1 vs C2
thresholds[4][0] = 0.5     # C2 vs A
thresholds[4][1] = 0.5     # C2 vs B
thresholds[4][2] = 0.5     # C2 vs C
thresholds[4][3] = 0.5     # C2 vs C1

# Process predictions to create confidence scores
max_index = []
confidences = []

n_samples = len(test_images)
logger.info("computing matrix with the predictions for each dynamic spectra...")
for num in range(n_samples):
    test_matrix = conf_matrix_from_pred(predictions, num)
    test_matrix -= thresholds
    mean_rows = np.nansum(test_matrix, axis=1)

    if np.all(np.isnan(mean_rows)):
        max_index.append(np.nan)
    else:
        max_index.append(int(np.nanargmax(mean_rows)))

    confidences.append(mean_rows.astype(np.float32))

logger.info("Saving the confidence scores...")
confidences = (
    np.vstack(confidences)
    if len(confidences)
    else np.empty((0, 5), dtype=np.float32)
)
logger.info("writing out the confidence scores on 2-D dynamic spectra to 'all_scores.npy' and classification results to 'All_scores.txt'.")
# Save confidence scores
np.save(OUT_DIR / "all_scores.npy", confidences)
logger.info("Confidence scores saved to 'all_scores.npy'.")

# Write classification results
out_path = OUT_DIR / "All_scores.txt"
mapping = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V"}

with out_path.open("w", encoding="utf-8") as f:
    for val in max_index:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            f.write("Unknown\n")
            continue
        f.write(f"{mapping.get(int(val), 'Unknown')}\n")

logger.info("Classification results saved to 'All_scores.txt'.")
