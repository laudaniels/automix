import os
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Toggle for enabling/disabling W&B logging (set AUTOMIX_WANDB_LOGS=1 to enable)
WANDB_LOGS = os.environ.get("AUTOMIX_WANDB_LOGS", "0") == "1"

# Define Training Parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 250
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
INTERVAL_LENGTH = 32
MASK_LENGTH = 2
SAMPLE_RATE = 50
FILE_PATH = os.environ.get("AUTOMIX_DATA_PATH", os.path.join(REPO_ROOT, "data", "processed-tokens"))

# Normalization statistics
EPSILON = 1e-6
VOCAB_SIZE = 1024

# Model Initialization
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 2048
MAX_SEQ_LENGTH = 1600
DROPOUT = 0.2

#Mamba Hyperparams
MAMBA_D_STATE=2
MAMBA_D_CONV = 2
MAMBA_EXPAND = 2