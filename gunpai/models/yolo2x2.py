import cv2
import torch
import coremltools as ct

import numpy as np
from ultralytics import YOLO
from PIL import Image
import subprocess

def start_yolo():

    # Load the YOLOv8 model
    # model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # Choose your desired YOLOv8 variant

    #model = YOLO("yolov8n-seg.pt")
    #model = YOLO("yolov8m.pt")
    # model = YOLO("yolov8n-seg.onnx")
    # model = YOLO("yolov8n-seg.pt")
    
    
    model = YOLO("yolov8s.pt")


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    output_rtsp_url = "rtsp://localhost:3554/yolo"
    output_rtmp_url = "rtmp://localhost:1935/stream/yolo"
    

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # Enable GPU acceleration if available
    if torch.cuda.is_available():
        model.cuda() 
    
    device = "mps"
    
    

    skip_frames = 2  # Process every 5th frame

    frame_count = 0  # Initialize frame count
    
    
    rtsp_url = 'rtsp://127.0.0.1:8554/cam1'


    # Define your RTSP stream URLs
    rtsp_urls = [
        'rtsp://127.0.0.1:8554/cam1',
        'rtsp://127.0.0.1:8554/cam2',
        'rtsp://127.0.0.1:8554/cam3',
        'rtsp://127.0.0.1:8554/cam4',
    ]
    caps = [cv2.VideoCapture(rtsp_url) for rtsp_url in rtsp_urls]
# Check if all streams are opened successfully
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Unable to open the RTSP stream {i + 1}.")
            exit()
    # rtsp_url = 'rtsp://127.0.0.1:8554/cam1'
    # cap = cv2.VideoCapture(rtsp_url)
    # Open the video capture
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify video file path
   
   
    # option2  
    #cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/bodycam")
    
    
    # blue_frame = np.full((320, 320, 3), (255, 0, 0), dtype=np.uint8)
    blue_frame = np.full((1280, 720, 3), (255, 0, 0), dtype=np.uint8)
    
    # cap = caps[0]
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    width = 1280
    height = 720
    fps = 15
    width = 1920
    height = 1080
    fps = 15
    

    # FFmpeg command to stream to RTSP
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{width}x{height}",
        '-r', str(15),
        '-i', '-',  # Input comes from the standard input
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-f', 'rtsp',
        '-rtsp_transport','tcp',
        output_rtsp_url
    ]
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{width}x{height}",
        '-r', str(15),
        '-i', '-',  # Input comes from the standard input
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p',
        '-f', 'flv',
        output_rtmp_url
    ]
    # Start FFmpeg process
    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    

    while(True):
        # Read a frame from the video
        # ret, frame = cap.read()
        frames = []
        # for cap in caps:
   #          ret, frame = cap.read()
   #          if not ret:
   #              print("Warning: Unable to read frame from one of the streams. Using blue frame.")
   #              frame = blue_frame  # Use the blue frame if the stream is not ready
   #          else:
   #              # frame = cv2.resize(frame, (1280  , 720))  # Resize to fit the grid
   #              ""
   #
   #          frames.append(frame)
            
        # ret, frame = caps[0].read()
        ret, frame = cap.read()
        
        
        # frames.append(frame)
        # frames.append(frame)
        # frames.append(frame)
        # frames.append(frame)
        #
        #
        # # Combine the frames into a single canvas (2x2 grid)
        # top_row = np.hstack(frames[0:2])  # Combine first two frames horizontally
        # bottom_row = np.hstack(frames[2:4])  # Combine last two frames horizontally
        # # top_row_2 = np.hstack(frames[8:12])  # Combine first two frames horizontally
        # # bottom_row_2 = np.hstack(frames[12:16])  # Combine last two frames horizontally
        #
        # canvas = np.vstack((top_row, bottom_row))  # Combine the two rows vertically
        frame_count += 1 
        if frame_count % skip_frames != 0:
            continue
        
        canvas = frame
        
        # for core ml
        
        # # Preprocess the frame (resize and normalize)
        # resized_frame = cv2.resize(canvas, (640, 640))  # Resize to match model input shape
        # normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
        #
        # # Convert to Core ML compatible format (CVPixelBuffer)
        # # image_input = ct.ImageType(name="image", shape=(1, 3, 640, 640))
        #
        # # Convert the NumPy array to a CVPixelBuffer
        # # coreml_input = image_input.from_numpy_array(normalized_frame)
        # pil_image = Image.fromarray(np.uint8(normalized_frame * 255))  # Convert to PIL Image
    
        # coreml_input = ct.pixel_buffer_from_numpy_array(normalized_frame)
    
        # try:
   #          output = model.predict({'image': pil_image})
   #      except Exception as e:
   #          print(f"Error during prediction: {e}")
   #          continue
   #
        results = model(canvas,device=device)


        annotated_frame = results[0].plot()
    #
    #
    #     output_frame = cv2.resize(annotated_frame, (1920  , 1080))
        
        # output_frame = cv2.resize(canvas, (1920  , 1080))
        
        output_frame = annotated_frame
        
        
        
        
    
        try:
            ffmpeg_process.stdin.write(output_frame.tobytes())
        except KeyboardInterrupt:
            print("Streaming stopped.")
     
        
     
    
   
        # Visualize the results on the frame
    

        # Display the annotated frame
        # cv2.imshow('YOLOv8 Object Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
