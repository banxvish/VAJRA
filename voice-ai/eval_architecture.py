import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

model = ptcv_get_model("xception", pretrained=False)
for name, module in model.named_modules():
    print(name)
