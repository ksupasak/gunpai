import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use "yolov8s.pt", "yolov8m.pt", etc.

# RTSP Stream URL (Replace with your camera details)
RTSP_URL = "rtsp://127.0.0.1:3554/live"

# Open RTSP Stream
cap = cv2.VideoCapture(RTSP_URL)
print('start')

if not cap.isOpened():
    print("Error: Couldn't open the RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = float(box.conf[0])  # Get confidence score
            class_id = int(box.cls[0])  # Get class ID
            label = f"{model.names[class_id]} {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8 RTSP", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()