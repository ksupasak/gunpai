import coremltools as ct

# Example: load ONNX and convert
import onnx
onnx_model = onnx.load("yolov8n.onnx")
mlmodel = ct.convert(onnx_model)
mlmodel.save("yolov8n.mlmodel")