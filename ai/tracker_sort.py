from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort  # Import the SORT tracking algorithm
# from sort.tracker import SortTracker
# Load YOLOv8 model
# model = YOLO('yolov8n.pt')  # Replace with your model path

# Initialize DeepSORT tracker
# tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
# Load an official or custom model
model = YOLO("yolo11n.pt")  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model
# Open video
# cap = cv2.VideoCapture(1)  

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')  # Choose a lightweight model for real-time tracking

# Initialize SORT tracker
tracker = Sort(max_age=100, min_hits=10, iou_threshold=0.5)

# Video Capture
# video_path = "video.mp4"  # Replace with your video path
# cap = cv2.VideoCapture(video_path)

cap = cv2.VideoCapture(2)  # Use 0 for default camera, or specify video file path



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the current frame
    results = model(frame, stream=True)

    # Prepare detections for SORT (format: [x1, y1, x2, y2, confidence])
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            detections.append([x1, y1, x2, y2, conf])

    # Convert to numpy array (SORT requires numpy input)
    detections = np.array(detections)
    
    try:
    # Update tracker with detections
        tracked_objects = tracker.update(detections)
      
        # Draw bounding boxes and track IDs
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    except Exception as e:
        print(f"Caught an error: {e}")
    # Display the frame
    cv2.imshow("YOLOv8 Object Tracking with SORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()