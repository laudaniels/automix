import os
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Set to "1" to enable W&B logging (requires `wandb login`)
WANDB_LOGS = os.environ.get("AUTOMIX_WANDB_LOGS", "0") == "1"

#data information
VOCAB_SIZE = 1024
AUDIO_CHANNELS = 4
DATA_PATH = os.environ.get("AUTOMIX_DATA_PATH", os.path.join(REPO_ROOT, "data", "processed-tokens"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INTERVAL_LENGTH = 32
MASK_LENGTH = 2
SAMPLE_RATE = 50

#architectural params
D_MODEL = 512 # 512
NUM_LAYERS = 5 # 5
NUM_HEADS = 16 #16
D_FF = 2048 # 2048
MAX_SEQ_LENGTH = 1600
DROPOUT = 0.15

#hyperparams
BATCH_SIZE = 2
EPOCHS = 1000
LEARNING_RATE = 5e-5

# Checkpoint to resume from (relative to the repo so it works on any machine)
CHECKPOINT_PATH = os.environ.get(
    "AUTOMIX_CHECKPOINT_PATH",
    os.path.join(REPO_ROOT, "runs", "transformer_runs", "bert_epoch_135.pt"),
)
START_EPOCH = int(os.environ.get("AUTOMIX_START_EPOCH", "135"))
