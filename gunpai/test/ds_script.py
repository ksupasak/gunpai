import cv2
import threading
import queue
import time
import torch
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

import numpy as np
from ultralytics import YOLO
from math import sqrt


# Configuration
NUM_STREAMS = 16
# RTSP_SOURCES = [f"rtsp://your_source_{i}" for i in range(NUM_STREAMS)]  # Replace with your actual RTSP URLs
RTSP_SOURCES = [1 for i in range(NUM_STREAMS)]  # Replace with your actual RTSP URLs
rtsp = 'rtsp://admin:DtFortune3@192.168.0.9:554/Streaming/channels/101'

rtsp = 'rtsp://127.0.0.1:8554/cam1'
rtsp = 'rtsp://127.0.0.1:8554/cam3?rtsp_transport=tcp'

# rtsp = 'rtsp://127.0.0.1:3554/live'

RTSP_SOURCES = [rtsp for i in range(NUM_STREAMS)]

RTSP_SOURCES[0] = 1

RTSP_SOURCES = [1 for i in range(NUM_STREAMS)]


print(RTSP_SOURCES)


# RESIZE_WIDTH, RESIZE_HEIGHT = int(1280), int(720)  # HD720 resolution
RESIZE_WIDTH, RESIZE_HEIGHT = int(1920/2), int(1080/2)  # HD720 resolution
RESIZE_WIDTH, RESIZE_HEIGHT = 416, 416  # HD720 resolution


# RESIZE_WIDTH, RESIZE_HEIGHT = int(1280), int(720)  # HD720 resolution

# Queues for inter-thread communication
BUFFER_SIZE = 10
motion_detection_queues = [queue.Queue(maxsize=BUFFER_SIZE) for _ in range(NUM_STREAMS)]
processing_queues = [queue.Queue(maxsize=BUFFER_SIZE) for _ in range(NUM_STREAMS)]
display_queues = [queue.Queue(maxsize=BUFFER_SIZE) for _ in range(NUM_STREAMS)]
capture_queues = [queue.Queue(maxsize=BUFFER_SIZE) for _ in range(NUM_STREAMS)]

# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"
# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt").to(device)  # Replace with your preferred model

# Shared flags for motion detection
motion_flags = [False] * NUM_STREAMS
lock = threading.Lock()

    # Create a grid for displaying all streams
grid_cols = int(sqrt(NUM_STREAMS))   
grid_rows = int(sqrt(NUM_STREAMS))  
grid_width = RESIZE_WIDTH // 2
grid_height = RESIZE_HEIGHT // 2
grid_width = int(1920/2)
grid_height = int(1080/2)
grid = np.zeros((grid_rows * grid_height, grid_cols * grid_width, 3), dtype=np.uint8)

# สร้าง thread pool executor
yolo_executor = ThreadPoolExecutor(max_workers=4)


FPS = 25
mode = 3


SAMPLE_INTERVAL = 1/FPS  # 1/10 second sampling for motion detection
MOTION_THRESHOLD = 500  # Adjust based on your needs



def capture_frames(stream_id):

    cap = None
    if(isinstance(RTSP_SOURCES[stream_id], (int, float))):
        cap = cv2.VideoCapture(RTSP_SOURCES[stream_id])
    else:
        cap = cv2.VideoCapture(RTSP_SOURCES[stream_id],cv2.CAP_FFMPEG)

    print(cv2.getBuildInformation())
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # cap.set(cv2.CAP_PROP_FPS, 5)   
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer size

    # cap = None
    # while True:
    #     time.sleep(0.1)
    #     cap = caps[stream_id]
    #     if cap is None:
    #         continue
    # print(f"Capture {stream_id} .. start")

    last_sample_time = time.time()
    while True:
        current_time = time.time()
        if current_time - last_sample_time < SAMPLE_INTERVAL:
            time.sleep(0.01)
            cap.grab()
            continue
        
        last_sample_time = current_time
        time.sleep(0.001)
        ret, frame = cap.read()
        # print(f"Capture frame {cv2.CAP_PROP_FPS} {stream_id} size: {frame.shape}")
        # print(f"Capture frame {stream_id} size: {frame.shape}")
        if not ret:
            print(f"Error reading frame from stream {stream_id}, reconnecting...")
            cap.release()
            time.sleep(5)  # Wait before reconnecting
            if(isinstance(RTSP_SOURCES[stream_id], (int, float))):
                cap = cv2.VideoCapture(RTSP_SOURCES[stream_id])
            else:
                cap = cv2.VideoCapture(RTSP_SOURCES[stream_id],cv2.CAP_FFMPEG)
            continue
        
        # Put frame in capture queue if there's space
        try:
            if capture_queues[stream_id].qsize() < BUFFER_SIZE:
                # print(f"Caputure queue {stream_id} size: {capture_queues[stream_id].qsize()}")

                capture_queues[stream_id].put(frame, timeout=0.1)
            else:
                # Drop frame if queue is full
                pass
        except queue.Full:
            pass

def motion_detection(stream_id):
    prev_frame = None
    prev_gray = None
    motion_detected = False
    global mode
    
    while True:
       
        
        # Get latest frame from capture queue
        try:
            # print(f"Motion queue {stream_id} size: {capture_queues[stream_id].qsize()}")
            time.sleep(0.001)
            frame = capture_queues[stream_id].get()
            while not capture_queues[stream_id].empty():
                capture_queues[stream_id].get()

            frame = cv2.resize(frame, (grid_width, grid_height))


            if(mode==1):
                display_queues[stream_id].put(frame)
            elif(mode==2):
                # frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
                                # frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

                processing_queues[stream_id].put(frame) 
       
            elif(mode==3 or mode==4):
              
                # frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

                # frame_resized = cv2.resize(frame, (grid_width, grid_height))
                frame_resized = frame
                
                # Initialize previous frame if first run
                if prev_frame is None:
                    prev_frame = frame_resized
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
                    continue
                
                # Motion detection processing
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                # frame_delta = cv2.absdiff(prev_gray, gray)
                # thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                # thresh = cv2.dilate(thresh, None, iterations=2)
                
                # contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # motion_detected = any(cv2.contourArea(c) > MOTION_THRESHOLD for c in contours)
                

                diff = cv2.absdiff(prev_gray, gray)

                # Apply threshold
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

                # Count nonzero pixels to detect motion
                motion_pixels = cv2.countNonZero(thresh)

                if motion_pixels > 500:
                    motion_detected = True
                else:
                    motion_detected = False
                
                if(mode==4):
                    motion_detected = False

                if( motion_detected ):
                    # print(f"Motion detected in stream {stream_id} {motion_pixels}")
                    processing_queues[stream_id].put(frame_resized)
                else:
                    # print(f"No motion detected in stream {stream_id}")
                    display_queues[stream_id].put(frame_resized)
                # with lock:
                #     if motion_detected and not motion_flags[stream_id] and stream_id%2==0:
                #         motion_flags[stream_id] = True
                #         if motion_detection_queues[stream_id].empty():
                #             processing_queues[stream_id].put(frame_resized)
                #     elif not motion_detected and motion_flags[stream_id]:
                #         motion_flags[stream_id] = False
                #     else:
                #         display_queues[stream_id].put(frame_resized)
                
                prev_gray = gray.copy()
            capture_queues[stream_id].task_done()
        except queue.Empty:
            # print(f"Motion queue {stream_id} is empty")
            time.sleep(0.01)
            continue
            

def live_processing(stream_id):
    while True:
        time.sleep(0.001)
        with lock:
            if motion_flags[stream_id]:
                try:
                    frame = motion_detection_queues[stream_id].get_nowait()
                    if processing_queues[stream_id].empty():
                        processing_queues[stream_id].put(frame)
                except queue.Empty:
                    pass
        
def process_frame(stream_id,frame):
    print("processing frame")
    results = yolo_model(frame, verbose=False)
    annotated_frame = results[0].plot()
    display_queues[stream_id].put(frame)
    print("frame processed")
    

# This will be our task queue with limited size
task_queue = queue.Queue(maxsize=NUM_STREAMS)  # max 10 tasks in queue
frames = [None for _ in range(NUM_STREAMS)]
# Function to submit jobs from queue to the executor
def queue_dispatcher(executor):
    while True:
        time.sleep(0.001)
        print(f"[{task_queue.qsize()}]")
        task = task_queue.get()
     
        # executor.submit(task[0], task[1])   
        frame = frames[task[1]]
        results = yolo_model(frame, verbose=False)
        frame = results[0].plot()
        display_queues[task[1]].put(frame)
        task_queue.task_done()    


def yolo_processing(stream_id,executor):
    while True:
        time.sleep(0.01)
        try:
            
            
          
            frame = processing_queues[stream_id].get()
            # print(f"[{stream_id}]  Processing frame {frame.shape} {capture_queues[stream_id].qsize()} {processing_queues[stream_id].qsize()} {display_queues[stream_id].qsize()} {task_queue.qsize()}")
            while not processing_queues[stream_id].empty():
                frame = processing_queues[stream_id].get()
      

            frames[stream_id] = frame
            # task_queue.put((process_frame, (stream_id)))
            print(f"[{stream_id}] Processing frame {frame.shape}")
            if(True):
                results = yolo_model(frame, verbose=False)
                frame = results[0].plot()
                display_queues[stream_id].put(frame)
            else:
                display_queues[stream_id].put(frame)
            processing_queues[stream_id].task_done()

            # display_queues[stream_id].put(frame)
            # if(task_queue.qsize() < 10):
            #     task_queue.put((process_frame, (stream_id,frame)))
                # print(f"[{stream_id}] Queueing task {task_queue.qsize()}")
            # else:
                # print(f"[{stream_id}] Queue is full {task_queue.qsize()}")

            # executor.submit(process_frame, stream_id,frame)
            # result = future.result()
        
                
        except queue.Empty:
            continue
       

def display_frames(stream_id):

    
    # Initialize grid with black frames
    
    last_frames = [None] * NUM_STREAMS
    
    while True:
        time.sleep(0.01)
        try:
            frame = display_queues[stream_id].get()
            while not display_queues[stream_id].empty():
                display_queues[stream_id].get()
            
            # last_frames[stream_id] = cv2.resize(frame, (grid_width, grid_height))
            
            last_frames[stream_id] = frame

            row = stream_id // grid_cols
            col = stream_id % grid_cols
            grid[row*grid_height:(row+1)*grid_height, col*grid_width:(col+1)*grid_width] = last_frames[stream_id]
            display_queues[stream_id].task_done()
            
        except queue.Empty:
            pass


def main():

    global mode
    global FPS
    global SAMPLE_INTERVAL

    print("Hello, World!")
    motion_threads = []
  
    capture_threads = []
    # Use ThreadPoolExecutor to initialize in parallel
   
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     results = executor.map(init_camera, RTSP_SOURCES)

    # for cap in results:
    #     if cap:
    #         caps.append(cap)
    
    for i in range(NUM_STREAMS):
        t = threading.Thread(target=capture_frames, args=(i,), daemon=True)
        t.start()
        capture_threads.append(t)
        # Create and start live processing threads

    for i in range(NUM_STREAMS):
        t = threading.Thread(target=motion_detection, args=(i,), daemon=True)
        t.start()
        motion_threads.append(t)


    display_threads = []
    for i in range(NUM_STREAMS):
        t = threading.Thread(target=display_frames, args=(i,), daemon=True)
        t.start()
        display_threads.append(t)



    # Create and start YOLO processing threads
    executor = ThreadPoolExecutor(max_workers=4)

    dispatcher_thread = threading.Thread(target=queue_dispatcher, args=(executor,))
    dispatcher_thread.start()

    yolo_threads = []
    for i in range(NUM_STREAMS):
        t = threading.Thread(target=yolo_processing, args=(i,executor,))
        t.start()
        yolo_threads.append(t)

    
    # for t in yolo_threads:
    #     t.join()
    
    # # Wait for display thread to finish


    while True:
        time.sleep(0.001)
        # Display the grid
       
        # position = 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)  # Green
        thickness = 2

        mode_text = "pass"
        if mode==2 :
            mode_text = "AI all"
        elif mode==3:
            mode_text = "AI after Motion"
        elif mode==4:
            mode_text = "Motion only"
        
        cv2.putText(grid, mode_text, (50, 100), font, font_scale, color, thickness, cv2.LINE_AA)
        
        fps_text = f"FPS: {FPS}"
        
        
        cv2.putText(grid, fps_text, (50, 200), font, font_scale, color, thickness, cv2.LINE_AA)
        




        cv2.imshow("RTSP Streams Grid", grid)
        

        k = cv2.waitKey(1) 
        

     
        if k & 0xFF == ord('m'):
            mode = (mode+1)%5
            if mode==0 :
                mode = 1 
       
        if k &0xFF == ord('='):
            FPS = FPS + 1
        elif k &0xFF == ord('-'):
            FPS = FPS - 1

        SAMPLE_INTERVAL = 1/FPS    

        # print(f"Mode changed to {mode} FPS: {FPS} SAMPLE_INTERVAL: {SAMPLE_INTERVAL}")

        if k & 0xFF == ord('q'):
            break
            
        

    
    cv2.destroyAllWindows()
    

  

if __name__ == "__main__":
    main()





