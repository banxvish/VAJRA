import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model

class VideoDeepfakeModel(nn.Module):
    def __init__(self):
        super(VideoDeepfakeModel, self).__init__()
        base_model = ptcv_get_model("xception", pretrained=False)
        self.base = nn.Sequential(base_model.features)
        # Custom head from the state dict keys we saw:
        class Head(nn.Module):
            def __init__(self):
                super(Head, self).__init__()
                self.b1 = nn.BatchNorm1d(2048)
                self.l = nn.Linear(2048, 512)
                self.b2 = nn.BatchNorm1d(512)
                self.o = nn.Linear(512, 1)
                self.relu = nn.ReLU()
            def forward(self, x):
                x = self.b1(x)
                x = self.l(x)
                x = self.relu(x)
                x = self.b2(x)
                x = self.o(x)
                return x
        # Wait, the key is `h1`, we'll just name it h1
        self.h1 = Head()

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.h1(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    model = VideoDeepfakeModel()
    model_path = r"c:\Users\bavis\OneDrive\Documents\VAJRA\voice-ai\video_pre trained\model.pth"
    # Filter out mismatch if we mapped base differently
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print("Video model loaded successfully!")
