from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('models/yolov8s.pt')

# Export to ONNX, OpenVINO, or TensorFlow Lite
model.export(format='openvino')  # or 'openvino', 'tflite'