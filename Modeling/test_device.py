import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU ({torch.cuda.get_device_name(0)}) is available.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

print(f"Using device: {device}")