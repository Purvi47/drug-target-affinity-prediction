import torch

checkpoint_path = "outputs/models/best_model.pt"

# Load the checkpoint dictionary safely
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Print all keys inside checkpoint
print("Keys inside checkpoint:")
print(checkpoint.keys())
