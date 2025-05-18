from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # หรือ yolov8s, yolov8m, etc.
model.export(format="onnx")
