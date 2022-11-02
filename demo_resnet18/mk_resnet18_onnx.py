import os
import torch
import torchvision.models as models

model = models.resnet18(weights='DEFAULT')
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

os.makedirs("data", exist_ok=True)
torch.onnx.export(model,
   dummy_input,
   "data/resnet18.onnx",
   verbose=False,
   input_names=["input.0"],
   output_names=["output.0"],
   export_params=True
   )
