import cv2
import threading
import queue
import time
import torch
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ultralytics import YOLO



# Configuration
NUM_STREAMS = 4
# RTSP_SOURCES = [f"rtsp://your_source_{i}" for i in range(NUM_STREAMS)]  # Replace with your actual RTSP URLs
RTSP_SOURCES = [1 for i in range(NUM_STREAMS)]  # Replace with your actual RTSP URLs
rtsp = 'rtsp://admin:DtFortune3@192.168.0.9:554/Streaming/channels/101'

rtsp = 'rtsp://127.0.0.1:8554/cam1'
rtsp = 'rtsp://127.0.0.1:8554/cam3'

# rtsp = 'rtsp://127.0.0.1:3554/live'

# RTSP_SOURCES = [rtsp for i in range(NUM_STREAMS)]



print(RTSP_SOURCES)


SAMPLE_INTERVAL = 0.01  # 1/10 second sampling for motion detection
MOTION_THRESHOLD = 500  # Adjust based on your needs
# RESIZE_WIDTH, RESIZE_HEIGHT = int(1280), int(720)  # HD720 resolution
RESIZE_WIDTH, RESIZE_HEIGHT = int(1920/4), int(1080/4)  # HD720 resolution

# RESIZE_WIDTH, RESIZE_HEIGHT = int(1280), int(720)  # HD720 resolution

# Queues for inter-thread communication
BUFFER_SIZE = 5
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
grid_cols = 2   
grid_rows = 2
grid_width = RESIZE_WIDTH // 2
grid_height = RESIZE_HEIGHT // 2
grid = np.zeros((grid_rows * grid_height, grid_cols * grid_width, 3), dtype=np.uint8)

# สร้าง thread pool executor
yolo_executor = ThreadPoolExecutor(max_workers=4)

def capture_frames(stream_id):
    cap = cv2.VideoCapture(RTSP_SOURCES[stream_id])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)   
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer size
    
    while True:
        time.sleep(0.01)
        ret, frame = cap.read()
        # print(f"Capture frame {stream_id} size: {frame.shape}")
        if not ret:
            print(f"Error reading frame from stream {stream_id}, reconnecting...")
            cap.release()
            time.sleep(5)  # Wait before reconnecting
            cap = cv2.VideoCapture(RTSP_SOURCES[stream_id])
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
    last_sample_time = 0
    
    while True:
        current_time = time.time()
        if current_time - last_sample_time < SAMPLE_INTERVAL:
            time.sleep(0.0001)
            continue
        
        last_sample_time = current_time
        
        # Get latest frame from capture queue
        try:
            # print(f"Motion queue {stream_id} size: {capture_queues[stream_id].qsize()}")
            time.sleep(0.001)
            frame = capture_queues[stream_id].get()
            while not capture_queues[stream_id].empty():
                capture_queues[stream_id].get()


            if(True):
                display_queues[stream_id].put(frame)
            elif(False):
                processing_queues[stream_id].put(frame) 
       
            else:
              
                frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
                
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
                
                if( motion_detected):
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
        time.sleep(0.01)
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
    frame = results[0].plot()
    display_queues[stream_id].put(frame)
    print("frame processed")
    

# This will be our task queue with limited size
task_queue = queue.Queue(maxsize=10)  # max 10 tasks in queue

# Function to submit jobs from queue to the executor
def queue_dispatcher(executor):
    j = 0 
    while True:
        time.sleep(0.001)
        print(f"[{j}]")
       
        
        if(processing_queues[j].qsize() > 0):
            frame = processing_queues[j].get(timeout=1)
            while not processing_queues[j].empty():
                processing_queues[j].get()
            executor.submit(process_frame, j,frame)
         
        j+=1

        j=0 if j>=NUM_STREAMS else j

        # task_queue.task_done()

def yolo_processing(stream_id,executor):
    while True:
        time.sleep(0.001)
        try:
            print(f"[{stream_id}] Processing frame {capture_queues[stream_id].qsize()} {processing_queues[stream_id].qsize()} {display_queues[stream_id].qsize()}")

            frame = processing_queues[stream_id].get()
            while not processing_queues[stream_id].empty():
                frame = processing_queues[stream_id].get()

            if(True):
                results = yolo_model(frame, verbose=True)
                # frame = results[0].plot()
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
        time.sleep(0.001)
        try:
            frame = display_queues[stream_id].get()
            while not display_queues[stream_id].empty():
                display_queues[stream_id].get()
            last_frames[stream_id] = cv2.resize(frame, (grid_width, grid_height))
            # last_frames[stream_id] = frame
            row = stream_id // grid_cols
            col = stream_id % grid_cols
            grid[row*grid_height:(row+1)*grid_height, col*grid_width:(col+1)*grid_width] = last_frames[stream_id]
            display_queues[stream_id].task_done()
            
        except queue.Empty:
            pass
        


def main():

    print("Hello, World!")
    motion_threads = []
  
    capture_threads = []
    
    
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

    # dispatcher_thread = threading.Thread(target=queue_dispatcher, args=(executor,))
    # dispatcher_thread.start()

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
        cv2.imshow("RTSP Streams Grid", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        

    
    cv2.destroyAllWindows()
    

    # cap = cv2.VideoCapture(1)


    # model = YOLO("yolov8n.pt")
    # while True:
    #     time.sleep(1)
    #     ret, frame = cap.read()
    #     results = model(frame, verbose=False)
    #     annotated_frame = results[0].plot()
    #     cv2.imshow("Frame", annotated_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()






# def live_processing(stream_id):
#     while True:
#         with lock:
#             if motion_flags[stream_id]:
#                 # Get the latest frame from motion detection
#                 try:
#                     frame = motion_detection_queues[stream_id].get_nowait()
#                     # Put frame in processing queue for YOLO
#                     if processing_queues[stream_id].empty():
#                         processing_queues[stream_id].put(frame)
#                 except queue.Empty:
#                     pass
        
#         time.sleep(0.01)  # Small sleep to prevent busy waiting

# def yolo_processing(stream_id):
#     while True:
#         try:
#             frame = processing_queues[stream_id].get(timeout=1)
            
#             # Run YOLOv8 inference
#             results = yolo_model(frame, verbose=False)
            
#             # Annotate frame with results
#             annotated_frame = results[0].plot()
            
#             # Put result in display queue
#             display_queue.put((stream_id, annotated_frame))
            
#         except queue.Empty:
#             continue

# def main():
#     # Create and start motion detection threads
#     motion_threads = []
#     for i in range(NUM_STREAMS):
#         t = threading.Thread(target=motion_detection, args=(i,), daemon=True)
#         t.start()
#         motion_threads.append(t)
    
#     # Create and start live processing threads
#     live_threads = []
#     for i in range(NUM_STREAMS):
#         t = threading.Thread(target=live_processing, args=(i,), daemon=True)
#         t.start()
#         live_threads.append(t)
    
#     # Create and start YOLO processing threads
#     yolo_threads = []
#     for i in range(NUM_STREAMS):
#         t = threading.Thread(target=yolo_processing, args=(i,), daemon=True)
#         t.start()
#         yolo_threads.append(t)
    
#     # Start display thread (non-daemon so it keeps the program running)
#     # display_thread = threading.Thread(target=display_frames)
#     # display_thread.start()
    
#     # # Wait for display thread to finish
#     # display_thread.join()

# if __name__ == "__main__":
#     main()