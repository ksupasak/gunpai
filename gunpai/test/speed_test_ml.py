import cv2
import threading
import queue
import time
import torch
from concurrent.futures import ThreadPoolExecutor
import coremltools as ct


import numpy as np
from ultralytics import YOLO
rtsp = 'rtsp://127.0.0.1:8554/cam3'
device = "mps" if torch.backends.mps.is_available() else "cpu"


cap = cv2.VideoCapture(1)  # Use 0 for webcam or provide video file path
#model = YOLO("yolov8n.pt").to(device)   
model = ct.models.MLModel("yolov8n.mlpackage")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    

    # Run YOLOv8 on the frame
    # results = model(frame)
    results = model.predict({"input": frame})
    

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"  # Label

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  


    end = time.time()

    print(f"Frame time: {(1/(end - start)):.2f} fps")

    annotated = results[0].plot()
    cv2.imshow("YOLOv8", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()