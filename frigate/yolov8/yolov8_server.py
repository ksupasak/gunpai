from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()
model = YOLO("yolov8n.pt")  # Load YOLOv8 model (you can change to other versions like yolov8s.pt)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Load image
    image = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Run YOLOv8 detection
    results = model(image)

    # Format results
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": box.cls.item(),
                "confidence": box.conf.item(),
                "box": box.xyxy.tolist()
            })

    return {"detections": detections}