from ultralytics import YOLO
from PIL import Image
import cv2


model = YOLO("yolo11n.mlpackage")  # Must export first
# img = Image.open("bus.jpg")
# results = model(img)
# print(results)


# Open default webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Annotate
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("YOLOv8 Webcam on macOS", annotated_frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()