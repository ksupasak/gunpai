# pip install ultralytics coremltools torch torchvision

from ultralytics import YOLO

import coremltools as ct

# import onnx
import onnx_coreml
# from onnx_coreml import convert


# Load the YOLOv8 model
model = YOLO("yolov8x.pt")  # Replace with your YOLOv8 model path

# Export to ONNX
model.export(format="onnx", opset=11)  # 'opset' may need adjustment for CoreML



# # Load the ONNX model
onnx_model_path = "yolov8x.onnx"  # Replace with your ONNX model path

# onnx_model = ct.utils.load_spec(onnx_model_path)


# coreml_model = ct.convert(onnx_model)#
#
# # coreml_model = ct.converters.convert(model=onnx_model_path)
# #
# # onnx_model = onnx.load(onnx_model_path)
# # coreml_model = convert(model=onnx_model)
#
# # import onnx_coreml
coreml_model = onnx_coreml.convert(onnx_model_path)
# #
# # coreml_model = convert(
# #     model=onnx_model_path,
# #     minimum_ios_deployment_target="13"  # Adjust based on your iOS target
# # )
# #
# # #
#
# # Save as .mlmodel
# coreml_model.save("yolov8x.mlmodel")
#
#
# from coremltools.models.neural_network import quantization_utils
#
# # Load the CoreML model
# coreml_model = ct.models.MLModel("yolov8x.mlmodel")
#
# # Quantize the model (optional, for smaller size and faster inference)
# quantized_model = quantization_utils.quantize_weights(coreml_model, nbits=16)
#
# # Save the quantized model
# quantized_model.save("yolov8x_quantized.mlmodel")