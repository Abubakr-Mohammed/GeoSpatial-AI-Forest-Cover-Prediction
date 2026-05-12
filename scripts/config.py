# config.py
# Central configuration for all project constants, paths, and hyperparameters.
# Edit this file to change behaviour without touching any other module.

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "cover_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
TARGET_COLUMN  = "class"
NUM_CLASSES    = 7          # Cover types 1-7 (remapped to 0-6 internally)
RANDOM_SEED    = 42

# Train / validation / test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ---------------------------------------------------------------------------
# Continuous features to be scaled (StandardScaler)
# ---------------------------------------------------------------------------
CONTINUOUS_FEATURES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

# ---------------------------------------------------------------------------
# Model architecture defaults
# ---------------------------------------------------------------------------
HIDDEN_UNITS   = [256, 128, 64]   # Neurons per hidden layer
DROPOUT_RATE   = 0.3
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 512
EPOCHS         = 50
EARLY_STOPPING_PATIENCE = 7

# ---------------------------------------------------------------------------
# Class names (index 0 = class label 1)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]