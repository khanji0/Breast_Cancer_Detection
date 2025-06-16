# File: src/config.py
# Description: Configuration settings for breast cancer detection project

import os

# Data paths
DATA_DIR = "data/archive"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"
EVALUATION_DIR = "evaluation_results"

# Data parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model parameters
MODEL_NAME = "resnet50"  # Options: resnet50, densenet121, efficientnet_b0
PRETRAINED = True
NUM_CLASSES = 2

# Training parameters
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 5
RANDOM_SEED = 42

# Data augmentation
TRAIN_TRANSFORM = {
    "RandomHorizontalFlip": True,
    "RandomVerticalFlip": True,
    "RandomRotation": 10,
    "ColorJitter": {
        "brightness": 0.2,
        "contrast": 0.2
    }
}

# Model architecture
MODEL_CONFIG = {
    "resnet50": {
        "fc_layers": [512, 2],
        "dropout": 0.3
    },
    "densenet121": {
        "fc_layers": [512, 2],
        "dropout": 0.3
    },
    "efficientnet_b0": {
        "fc_layers": [512, 2],
        "dropout": 0.3
    }
}

# Evaluation metrics
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "average_precision"
]

# Visualization settings
PLOT_SETTINGS = {
    "figsize": (12, 8),
    "dpi": 100,
    "style": "seaborn",
    "save_format": "png"
}

# Create necessary directories
for directory in [PROCESSED_DATA_DIR, MODEL_DIR, EVALUATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "training.log",
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
} 