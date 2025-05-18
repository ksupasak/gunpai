import cv2
import threading
import queue
import time
import torch
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ultralytics import YOLO
rtsp = 'rtsp://127.0.0.1:8554/cam3'
device = "cpu" # if torch.backends.mps.is_available() else "cpu"


cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
model = YOLO("yolov8n.pt").to(device)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    results = model(frame)
    end = time.time()

    print(f"Frame time: {(1/(end - start)):.2f} fps")

    annotated = results[0].plot()
    cv2.imshow("YOLOv8", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()