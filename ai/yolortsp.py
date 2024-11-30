import cv2
import torch



from ultralytics import YOLO


# Load the YOLOv8 model
# model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # Choose your desired YOLOv8 variant

#model = YOLO("yolov8n-seg.pt")
#model = YOLO("yolov8m.pt")
# model = YOLO("yolov8n-seg.onnx")
model = YOLO("yolov8n-seg.pt")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# face_cascade_gpu = cv2.cuda.CascadeClassifier_create(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# Enable GPU acceleration if available
if torch.cuda.is_available():
    model.cuda() 
    
device = "mps"

# Open the video capture
# cap = cv2.VideoCapture(1)  # Use 0 for default camera, or specify video file path
rtsp_url = 'rtsp://10.149.1.62:8554/test'
cap = cv2.VideoCapture(rtsp_url)

while(True):
    # Read a frame from the video
    ret, frame = cap.read()


    # if cuda #############################
    if(False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Convert to grayscale on the GPU
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

        # Download the grayscale image to the CPU
        gray = gpu_gray.download()
    
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    

    # Run inference on the frame
    # faces_gpu = face_cascade_gpu.detectMultiScale(gpu_gray)
    

    
    
    results = model(frame,device=device)
    annotated_frame = results[0].plot()
    
    if(False):
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Upload the frame to the GPU
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    
    # Visualize the results on the frame
    

    # Display the annotated frame
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()