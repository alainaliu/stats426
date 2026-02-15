# Paths
DATA_DIR       = "nabirds"
OUTPUT_DIR     = "checkpoints"

# Image / loader
IMG_SIZE       = 224  # standard ResNet input
BATCH_SIZE     = 32
NUM_WORKERS    = 4

# Training
EPOCHS         = 10   # short run to confirm loss decreases
LR             = 1e-3
BACKBONE_LR    = 1e-4
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.1
DROPOUT        = 0.3
WARMUP_EPOCHS  = 2
SEED           = 42

# Preprocessing
USE_BBOX       = True

# Subset control (Set USE_SUBSET = False to use all classes)
USE_SUBSET     = True    # True = classes 295-400 only (fast)
CLASS_MIN      = 295     # inclusive
CLASS_MAX      = 400     # inclusive

# Architecture
BACKBONE       = "resnet18"
