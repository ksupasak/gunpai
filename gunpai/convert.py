# from ultralytics import YOLO
# from onnx_coreml import convert
# import os

# # Step 1: Export YOLOv8 to ONNX
# model = YOLO("yolov8n.pt")
# model.export(format="onnx", opset=12, dynamic=False, simplify=True)

# # Step 2: Convert ONNX to CoreML
# onnx_path = "yolov8n.onnx"
# assert os.path.exists(onnx_path), f"ONNX file not found at {onnx_path}"

# coreml_model = convert(
#     model=onnx_path,
#     minimum_ios_deployment_target='13'
# )

# coreml_model.save("yolov8n.mlmodel")
# print("✅ CoreML model saved as yolov8n.mlmodel")


import coremltools as ct

# Convert from ONNX to Core ML
coreml_model = ct.converters.onnx.convert(model='yolov8n.onnx')
coreml_model.save('yolov8n.mlmodel')
print("✅ CoreML model saved using onnx-coreml")