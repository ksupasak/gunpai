import cv2
import numpy as np
import queue
import threading
import time
from ultralytics import YOLO

# RTSP Sources
rtsp_sources = [
 "rtsp://127.0.0.1:3554/live",
  "rtsp://127.0.0.1:3554/live", "rtsp://127.0.0.1:3554/live"
]

# Configuration
motion_threshold = 50000  # Motion detection threshold
inference_threads = 2  # Number of YOLO inference workers

# Queues
read_queue = queue.Queue(maxsize=50)
inference_queue = queue.Queue(maxsize=50)
render_queue = queue.Queue(maxsize=50)

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")


def capture_frames(rtsp_url, source_id):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Failed to open {rtsp_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read from {rtsp_url}")
            break

        timestamp = time.time()
        read_queue.put((source_id, frame, timestamp))
        time.sleep(0.03)  # Adjust capture rate

    cap.release()


def preprocess_frames(source_id):
    prev_frame = None

    while True:
        source, frame, timestamp = read_queue.get()
        if source != source_id:
            continue  # Ensure this thread processes its assigned source

        resized = cv2.resize(frame, (640, 480))

        motion_detected = False
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
            diff_score = np.sum(diff)
            if diff_score > motion_threshold:
                motion_detected = True

        prev_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if motion_detected:
            inference_queue.put((source, resized, timestamp))


def yolo_inference():
    while True:
        source, frame, timestamp = inference_queue.get()
        results = model(frame)

        # Extract detections
        detected_objects = results[0].boxes.xyxy.cpu().numpy()

        render_queue.put((source, frame, detected_objects, timestamp))


def render_output():
    while True:
        source, frame, detections, timestamp = render_queue.get()

        for box in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(f"Source {source}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# Start RTSP capture threads
for idx, url in enumerate(rtsp_sources):
    threading.Thread(target=capture_frames, args=(url, idx), daemon=True).start()
    threading.Thread(target=preprocess_frames, args=(idx,), daemon=True).start()

# Start inference threads
for _ in range(inference_threads):
    threading.Thread(target=yolo_inference, daemon=True).start()

# Start rendering thread
threading.Thread(target=render_output, daemon=True).start()

# Keep main thread alive
while True:
    time.sleep(1)