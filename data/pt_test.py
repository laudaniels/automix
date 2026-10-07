import os
import torch

file_path = os.path.join(
    os.environ.get("AUTOMIX_DATA_PATH", os.path.join(os.path.dirname(__file__), "processed-tokens")),
    "Don't_Stop_The_Music_-_Rihanna_encoded_codes.pt",
)
data = torch.load(file_path)  # Load the tokenized input

if not isinstance(data, torch.Tensor):
    raise ValueError(f"Loaded data is not a tensor! Got {type(data)}")

print(f"Loaded data shape: {data.shape}")  # Debugging info
print("Loaded data:", data)