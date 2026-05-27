from pathlib import Path

# ---------------
# Reproducibility
# ---------------

SEED = 42

# -----
# Model
# -----

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
HIDDEN_DIM = 256

# ------------------------
# Continual learning setup
# ------------------------

NUM_TASKS = 5
CLASSES_PER_TASK = 10
TOTAL_CLASSES = NUM_TASKS * CLASSES_PER_TASK

# --------
# Training
# --------

BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
EPOCHS_PER_TASK = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# --------------
# Replay memory
# --------------

MEMORY_SIZE = 500
HYBRID_RISK_FRACTIONS = [0.25, 0.50, 0.75]

# ---------------
# Risk estimation
# ---------------

PROBE_STEPS = 10
CORRECTED_PROBE_PROB_WEIGHT = 0.8
CORRECTED_PROBE_LOGIT_WEIGHT = 0.2

# -----
# Data
# -----

DATA_ROOT = Path("data")

# CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)