import cv2
import torch
import coremltools as ct

import numpy as np
from ultralytics import YOLO
from PIL import Image


# Load the YOLOv8 model
# model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # Choose your desired YOLOv8 variant

#model = YOLO("yolov8n-seg.pt")
#model = YOLO("yolov8m.pt")
# model = YOLO("yolov8n-seg.onnx")
model = YOLO("yolov8x.pt")


try:
    model = ct.models.MLModel('yolov8x.mlpackage')  # Replace with your Core ML model file
except Exception as e:
    print(f"Error loading Core ML model: {e}")
    exit()

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# Enable GPU acceleration if available
if torch.cuda.is_available():
    model.cuda() 
    
    

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# Enable GPU acceleration if available
if torch.cuda.is_available():
    model.cuda() 
    
device = "mps"

skip_frames = 5  # Process every 5th frame

frame_count = 0  # Initialize frame count


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
    rtsp_url
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
    
    # Only process every 'skip_frames' frame
    if frame_count % skip_frames == 0:
    
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Unable to read frame from one of the streams. Using blue frame.")
                frame = blue_frame  # Use the blue frame if the stream is not ready
            else:
                frame = cv2.resize(frame, (320  , 320))  # Resize to fit the grid
        
            frames.append(frame)

        # Combine the frames into a single canvas (2x2 grid)
        top_row = np.hstack(frames[0:2])  # Combine first two frames horizontally
        bottom_row = np.hstack(frames[2:4])  # Combine last two frames horizontally
        # top_row_2 = np.hstack(frames[8:12])  # Combine first two frames horizontally
        # bottom_row_2 = np.hstack(frames[12:16])  # Combine last two frames horizontally
    
        canvas = np.vstack((top_row, bottom_row))  # Combine the two rows vertically


     # Preprocess the frame (resize and normalize)
        resized_frame = cv2.resize(canvas, (640, 640))  # Resize to match model input shape
        normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]

        # Convert to Core ML compatible format (CVPixelBuffer)
        # image_input = ct.ImageType(name="image", shape=(1, 3, 640, 640))
    
        # Convert the NumPy array to a CVPixelBuffer
        # coreml_input = image_input.from_numpy_array(normalized_frame)
        pil_image = Image.fromarray(np.uint8(normalized_frame * 255))  # Convert to PIL Image
    
        # coreml_input = ct.pixel_buffer_from_numpy_array(normalized_frame)
    
        try:
            output = model.predict({'image': pil_image}) 
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue
        # print(output)
     # # Assuming the output is a dictionary with 'boxes', 'labels', and 'scores' keys
        boxes = output['coordinates']  # Array of bounding boxes (x1, y1, x2, y2)
        # labels = output['labels']  # Array of class labels (integers)
        # scores = output['scores']  # Array of confidence scores
        #
        # # Plot the bounding boxes and labels on the original frame
        # for box in zip(boxes):
        #     x1, y1, x2, y2 = map(int, box[0] * 640)  # Convert box coordinates to integers
        #
        #     cv2.rectangle(resized_frame, (x1-(int)(x2/2), y1-(int)(y2/2)), ((int)(x2/2)+x1, (int)(y2/2)+y1), (0, 255, 0), 2)  # Draw green rectangle
        # #     cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # # annotated_frame = output[0].plot()
        # # Display the annotated frame
        # cv2.imshow('YOLOv8 Object Detection', resized_frame)
        #


        ##################################################
        # results = model(canvas,device=device)
        # annotated_frame = results[0].plot()
        # # Display the combined frame
        # cv2.imshow('Combined RTSP Streams', annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_count += 1
# Release the captures and close windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()