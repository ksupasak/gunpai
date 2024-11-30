
import cv2
import torch



from ultralytics import YOLO


# Load the YOLOv8 model
# model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # Choose your desired YOLOv8 variant

#model = YOLO("yolov8n-seg.pt")
#model = YOLO("yolov8m.pt")
# model = YOLO("yolov8n-seg.onnx")
model = YOLO("yolov8x.pt")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# Enable GPU acceleration if available
if torch.cuda.is_available():
    model.cuda() 
    
device = "mps"



import cv2
import numpy as np

# rtsp_url = 'rtsp://10.149.1.62:8554/test'
# rtsp_url = 'rtsp://10.149.1.34:8554/cam1'
rtsp_url = 'rtsp://127.0.0.1:8554/cam1'


# Define your RTSP stream URLs
rtsp_urls = [
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    rtsp_url,
    
]
import cv2
import numpy as np

# Create VideoCapture objects for each stream
caps = [cv2.VideoCapture(rtsp_url) for rtsp_url in rtsp_urls]

# Check if all streams are opened successfully
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Unable to open the RTSP stream {i + 1}.")
        exit()

# Define a blue frame (960x540)
# blue_frame = np.full((540, 960, 3), (255, 0, 0), dtype=np.uint8)
blue_frame = np.full((320, 320, 3), (255, 0, 0), dtype=np.uint8)

while True:
    frames = []
    
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unable to read frame from one of the streams. Using blue frame.")
            frame = blue_frame  # Use the blue frame if the stream is not ready
        else:
            frame = cv2.resize(frame, (320  , 320))  # Resize to fit the grid
        
        frames.append(frame)

    # Combine the frames into a single canvas (2x2 grid)
    top_row = np.hstack(frames[0:4])  # Combine first two frames horizontally
    bottom_row = np.hstack(frames[4:8])  # Combine last two frames horizontally
    top_row_2 = np.hstack(frames[8:12])  # Combine first two frames horizontally
    bottom_row_2 = np.hstack(frames[12:16])  # Combine last two frames horizontally
    
    canvas = np.vstack((top_row, bottom_row,top_row_2, bottom_row_2))  # Combine the two rows vertically


    results = model(canvas,device=device)
  
    
    annotated_frame = results[0].plot()
 
    cv2.imshow('Combined RTSP Streams', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the captures and close windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()