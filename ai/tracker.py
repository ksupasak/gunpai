
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Load YOLOv8 model
# model = YOLO('yolov8n.pt')  # Replace with your model path

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
# Load an official or custom model
model = YOLO("yolo11n.pt")  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model
# Open video
cap = cv2.VideoCapture(1)  


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)
    # annotated_frame = results[0].plot()
    
    detections = []
    for box in results[0].boxes:
      confidence = float(box.conf[0])  # Convert confidence to a float
      class_id = int(box.cls[0])  # Convert class ID to an integer
      # box = box.xyxy[0].cpu().numpy()  # Convert to NumPy array
      # x1, y1, x2, y2 = box[:4]  # Extract bounding box coordinates
      x1, y1, x2, y2 = box.xyxy[0].tolist()
      print([x1, y1, x2, y2, confidence, class_id])
      detections.append([[x1, y1, x2, y2], confidence, class_id])
      
    print(detections)
        
    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # # Draw tracking results
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        l, t, r, b = track.to_tlbr()
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show video with tracking
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()