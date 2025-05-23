import cv2
from datetime import datetime
# HLS stream URL (must be public or authenticated properly)
url = 'https://streaming.udoncity.go.th:1935/live/Axis_IP754.stream/chunklist_w553778697.m3u8'

import cv2
from ultralytics import YOLO
import time

# Your HLS .m3u8 stream URL
HLS_URL = url

# Initialize YOLO model
model = YOLO("yolov8n.pt").to("mps")
model.fuse()

# Open video stream
cap = cv2.VideoCapture(HLS_URL)
if not cap.isOpened():
    print("Failed to open stream.")
    exit()

batch = []
batch_size = 16

now = datetime.now()
count = 0 
last = now

while True:
    ret, frame = cap.read()
    annotated = None

    count +=1
    nx = datetime.now()
    if (nx-last).seconds > 1:
        last = nx
        print(f"Time: {now} FPS: {count}")
        count = 0
    # every second
  

    if not ret:
        print("Stream lost or ended.")
        break

    # Resize for faster processing if needed (optional)
    # frame = cv2.resize(frame, (640, 640))
    batch.append(frame)

    if False and len(batch) >= batch_size:
        start = time.time()

        results = model(batch, imgsz=640, verbose=False)

        # for i, r in enumerate(results):
        #     annotated = r.plot()
        #     cv2.imshow(f"Batch Frame {i}", annotated)
        annotated = results[0].plot()
        end = time.time()
        print(f"Processed batch of {batch_size} in {end - start:.2f}s")

        batch.clear()
        

    # Allow quitting with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()