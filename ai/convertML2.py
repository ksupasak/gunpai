import torch
import coremltools as ct

# PyTorch to ONNX
model = torch.load('yolov8x.pt')
input_tensor = torch.randn(1, 3, 224, 224)  # Example input size
torch.onnx.export(model, input_tensor, "yolov8x.onnx")
#
# # ONNX to Core ML
# mlmodel = ct.converters.onnx.convert("yolov8x.onnx")
# mlmodel.save('yolov8x.mlmodel')