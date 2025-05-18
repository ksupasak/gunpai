import cv2
import torch
import time
import imageio.v3 as iio
import queue
import threading
from ultralytics import YOLO

# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"

# Load the YOLOv8 model
model = YOLO("yolov8x.pt").to(device)

# Open the video file
video_path = "../videos/cctv.mp4"

# Open the video stream
video_path = "rtsp://127.0.0.1:8554/cam4"

rtsp_url = video_path

# Load YOLOv8 model (runs on GPU if available)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# model = YOLO("yolov8n.pt").to(device)
model = YOLO("yolov8n.pt").to(device)

# model = YOLO("yolov8n-seg.pt").to(device)
print(rtsp_url)

# reader = iio.imiter(rtsp_url, plugin="pyav", format="rtsp", thread_type="AUTO")



# Video input and output settings
# video_path = "video.mp4"
# Open the video file
# video_path = "../videos/cctv.mp4"
video_path = 1 

# cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
#
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


cap = cv2.VideoCapture(video_path)


# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (, height))

# Thread-safe queues
frame_queue = queue.Queue(maxsize=1000)  # Holds raw frames
result_queue = queue.Queue(maxsize=1000)  # Holds processed frames

# Flag to indicate processing state
processing = True

ret, frame = cap.read()

results = model(frame)
print('Ready')    
    
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

motion_detect = False
# 🔹 Thread-1: Read video frames and put them in frame_queue
def read_frames():
    print('start read frame')
    # for frame in reader:
    #     frame_count += 1
    #     if frame_count % skip_frames != 0:
    #         continue  # Skip frames
    #
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("RTSP Stream", frame)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    while processing:
        # print('start read frame')
        skip_frames = 1  # Number of frames to skip
        # current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    #     print(current_frame)
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + skip_frames)

        ret, frame = cap.read()
        if not ret:
            break
        # print('write read frame ')
        frame_queue.put(frame)
    cap.release()


# 🔹 Thread-2: Process frames using YOLOv8 and put results in result_queue
def process_frames():
    print('start process ')
    
    while processing or not frame_queue.empty():
        
        
        if not frame_queue.empty():
            # print('start process frame x')
            print(f'write result {frame_queue.qsize()} {result_queue.qsize()}')
            
            frame = frame_queue.get()
            # result_queue.put(frame)
            
            # print('start process frame result')
            while not frame_queue.empty():
                frame_queue.get()


         # Convert to grayscale and apply background subtraction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(gray)

            # Find contours to detect motion
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = False

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Adjust sensitivity
                    motion_detected = True
                    break

            # Run YOLOv8 inference
            if motion_detected:
                motion_detect = True
                results = model(frame, device=device)

            #results = model(frame)
            # print('start process frame result')

            # Draw detections on the frame
            # for result in results:
          #       for box in result.boxes:
          #           x1, y1, x2, y2 = map(int, box.xyxy[0])
          #           conf = float(box.conf[0])
          #           cls = int(box.cls[0])
          #           label = f"{model.names[cls]} {conf:.2f}"
          #
          #           # Draw bounding box
          #           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
          #           cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #print(f'write result {frame_queue.qsize()} {result_queue.qsize()}')
            # result_queue.put(frame)

                frame = results[0].plot()
                result_queue.put(frame)
                
         


# 🔹 Thread-3: Display and save processed frames
def display_frames():
    print('xxxxx display')
    
    while processing or not result_queue.empty():
        if not result_queue.empty():
            print('start display')
            frame = result_queue.get()
            
            while not result_queueempty():
                result_queue.get()
            
            # out.write(frame)  # Save output video
            cv2.imshow("YOLOv8 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.01)

    # out.release()
    cv2.destroyAllWindows()

t2 = threading.Thread(target=process_frames)
# t3 = threading.Thread(target=display_frames)
# Start threads
t1 = threading.Thread(target=read_frames)

t2.start()

t1.start()

prev_time = cv2.getTickCount()
fps = 0


while processing or not result_queue.empty():
    if not result_queue.empty():
        print('start display')
        frame = result_queue.get()
        while not result_queue.empty():
            result_queue.get()
        # out.write(frame)  # Save output video
        # frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # Calculate FPS
        current_time = cv2.getTickCount()
        time_elapsed = (current_time - prev_time) / cv2.getTickFrequency()
        prev_time = current_time
        fps = 1 / time_elapsed if time_elapsed > 0 else 0

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       

        cv2.imshow("YOLOv8 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            processing = False
            break
    else:
        time.sleep(0.01)

# out.release()
cv2.destroyAllWindows()
# Wait for threads to complete
t1.join()
t2.join()
processing = False  # Signal processing should stop
