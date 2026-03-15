import torch
model_path = r"c:\Users\bavis\OneDrive\Documents\VAJRA\voice-ai\video_pre trained\model.pth"
checkpoint = torch.load(model_path, map_location="cpu")
print(type(checkpoint))
if isinstance(checkpoint, dict):
    print(checkpoint.keys())
    # If it is an OrderedDict of state dict:
    for k in list(checkpoint.keys())[-20:]:
        print(f"{k}: {checkpoint[k].shape}")
else:
    print(checkpoint)
