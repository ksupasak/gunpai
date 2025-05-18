import cv2
import torch
from ultralytics import YOLO
import coremltools as ct
import numpy as np


# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"

# Load the YOLOv8 model
# model = YOLO("yolov8n.mlpackage").to(device)

# Load Core ML model
model = ct.models.MLModel("yolov8n.mlpackage")


# Open the video file
video_path = "../videos/cctv.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
# out = cv2.VideoWriter("output_hd.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

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

    # Write the frame to output video
    # out.write(frame)

    # Show the frame
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()