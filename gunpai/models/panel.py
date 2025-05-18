import cv2
from sort import Sort  # Import the SORT tracking algorithm
import numpy as np
import threading
import time
import math
import traceback
import queue
from math import sqrt

from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from models.rule import Rule
        
class Panel:
    def __init__(self, id):
        self.id = id
        self.tracker = Sort(max_age=100, min_hits=10, iou_threshold=0.5)
        self.channels = []
        self.caps = []
        self.pre_records = []

        self.ready = False
        self.buffer_size = 10
        self.number_of_streams = 16
        self.sample_interval = 0.01
        self.resize_width = 1920
        self.resize_height = 1080

        
    def config(self, ctrl,  params):
        
        self.ctrl = ctrl
        print(f"params {params}")
        
        row = 4
        col = 4

        buffer_size = self.ctrl.fps * self.ctrl.pre_record

        for i, channel in  enumerate(params):
            
            ox = i%4 * int(ctrl.width/4)
            oy = i/4 * int(ctrl.height/4)

            
            channel.open(self, ox, oy)
            if(channel.ready==True):
                self.channels.append(channel)
                self.caps.append(channel.cap) 
        # self.number_of_streams = len(params)        
        self.motion_detection_queues = [queue.Queue(maxsize=self.buffer_size) for _ in range(len(params))]
        self.processing_queues = [queue.Queue(maxsize=self.buffer_size) for _ in range(len(params))]
        self.display_queues = [queue.Queue(maxsize=self.buffer_size) for _ in range(len(params))]
        self.capture_queues = [queue.Queue(maxsize=self.buffer_size) for _ in range(len(params))]
        self.pre_records = [deque(maxlen=buffer_size) for _ in range(len(params))]
        self.motion_flags = [False] * len(params)  
        self.grid_cols = int(sqrt(self.number_of_streams))   
        self.grid_rows = int(sqrt(self.number_of_streams))  
        self.grid_width = self.resize_width // 2
        self.grid_height = self.resize_height // 2
        self.last_frame = np.zeros((self.grid_rows * self.grid_height, self.grid_cols * self.grid_width, 3), dtype=np.uint8)

        self.ready=True
        self.task_queue = queue.Queue(maxsize=self.number_of_streams)  # max 10 tasks in queue
        self.frames = [None for _ in range(self.number_of_streams)]      
        self.sample_interval = 1/self.ctrl.fps
   
    def capture_frames(self, stream_id):

        cap = self.caps[stream_id]
        self.sample_interval = 1/self.ctrl.fps
      
        last_sample_time = time.time()
        while True:
            current_time = time.time()
            if current_time - last_sample_time < self.sample_interval:
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
                if(isinstance(self.caps[stream_id], (int, float))):
                    cap = cv2.VideoCapture(self.caps[stream_id])
                else:
                    cap = cv2.VideoCapture(self.caps[stream_id],cv2.CAP_FFMPEG)
                continue
            
            # Put frame in capture queue if there's space
            try:
                if self.capture_queues[stream_id].qsize() < self.buffer_size:
                    # print(f"Caputure queue {stream_id} size: {capture_queues[stream_id].qsize()}")

                    self.capture_queues[stream_id].put(frame, timeout=0.1)
                else:
                    # Drop frame if queue is full
                    pass
            except queue.Full:
                pass

    def motion_detection(self, stream_id):
        global detect_channel
        prev_frame = None
        prev_gray = None
        motion_detected = False
        global mode
        
        while True:
        
            
            # Get latest frame from capture queue
            try:
                # print(f"Motion queue {stream_id} size: {capture_queues[stream_id].qsize()}")
                time.sleep(0.001)
                frame = self.capture_queues[stream_id].get()
                while not self.capture_queues[stream_id].empty():
                    self.capture_queues[stream_id].get()

                frame = cv2.resize(frame, (self.grid_width, self.grid_height))


                if(self.ctrl.mode==1):
                    self.display_queues[stream_id].put(frame)
                elif(self.ctrl.mode==2):
                    if stream_id % (2**self.ctrl.detect_channel) == 0 :
                        self.processing_queues[stream_id].put(frame) 
                    else:
                        self.display_queues[stream_id].put(frame)
        
                elif(self.ctrl.mode==3 or self.ctrl.mode==4):
                
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
                    
                    if(self.ctrl.mode==4):
                        motion_detected = False
                    
                    if( motion_detected and stream_id %  (2**self.ctrl.detect_channel) == 0 ) :
                        # print(f"Motion detected in stream {stream_id} {motion_pixels}")
                        self.processing_queues[stream_id].put(frame_resized)
                    else:
                        # print(f"No motion detected in stream {stream_id}")
                        self.display_queues[stream_id].put(frame_resized)
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
                self.capture_queues[stream_id].task_done()
            except queue.Empty:
                # print(f"Motion queue {stream_id} is empty")
                time.sleep(0.01)
                continue
    def write_capture(self, stream_id):
        frame = self.frames[stream_id]
        print("Event triggered! Saving pre-recorded video...")

        filename = f"events/capture_{stream_id}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        return filename
    
    def write_pre_record(self, stream_id):
      
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame = self.frames[stream_id]
        print("Event triggered! Saving pre-recorded video...")

        filename = f"events/pre_record_{stream_id}_{int(time.time())}.mp4"
        out = cv2.VideoWriter(filename, fourcc, self.ctrl.fps, (frame.shape[1], frame.shape[0]))

        for buffered_frame in self.pre_records[stream_id]:
            out.write(buffered_frame)

        out.release()
        print(f"Saved: {filename}")
        return filename
        
    def object_detection(self, stream_id,executor):

        class_names = self.ctrl.model.names
        track_ids = {}
        track_live = {}
        track_loc_map ={}
        tracker =  Sort(max_age=100, min_hits=10, iou_threshold=0.5)   


        while True:
            time.sleep(0.01)
            try:
                             
            
                frame = self.processing_queues[stream_id].get()
                # print(f"[{stream_id}]  Processing frame {frame.shape} {self.capture_queues[stream_id].qsize()} {self.processing_queues[stream_id].qsize()} {self.display_queues[stream_id].qsize()} {self.  task_queue.qsize()}")
                while not self.processing_queues[stream_id].empty():
                    frame = self.processing_queues[stream_id].get()
        

                # self.frames[stream_id] = frame
                # task_queue.put((process_frame, (stream_id)))
                # print(f"[{stream_id}] Processing frame {frame.shape}")
                if(True):
                    results = self.ctrl.model(frame, verbose=False)
                    # frame = results[0].plot()
                else:
                    self.display_queues[stream_id].put(frame)

                self.processing_queues[stream_id].task_done()
                    
            # Extract detections from the first frame
                # detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class_id]
                res = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class_id]
            
                # detections = []
        #       for r in res:
        #           detections.append(r)

                # Convert to numpy array (SORT requires numpy input)
                detections = np.array(res)
            
                boxes = [det[:5] for det in detections]  # Extract [x1, y1, x2, y2, confidence]
                class_ids = [det[5] for det in detections]  # Extract class IDs

                # print(f"[{stream_id}] Detections {class_ids}")

                # Create a dictionary to associate track IDs with class IDs
                track_class_map = {}
        
                ch_map = {}
        
                
                # Update tracker with detections
                tracked_objects = tracker.update(detections)
    
                # tracked_objects frmat: [x1, y1, x2, y2, track_id]
                for i in detections:
                    key = f"{int(i[0])}-{int(i[1])}"
                    track_class_map[key] = int(i[5])
            
                
    
                # Draw bounding boxes and track IDs
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = map(int, obj)
                    # print(f"[{stream_id}] Tracking object {track_id} {x1} {y1} {x2} {y2}")
            
                    class_name = "-"
                    
                    if track_ids.get(track_id) is not None:
                        class_id = track_ids[track_id]
                        class_name = class_names[class_id]
                    else:
                
                        key = f"{int(x1)}-{int(y1)}"
                        if track_class_map.get(key) is not None :
                            class_id = int(track_class_map[key])
                            track_ids[track_id] = class_id
                    
                            class_name = class_names[class_id]
                        
                    speed = 0
                    xn = 0
                    yn = 0
                    if track_loc_map.get(track_id) is not None :
                        loc = track_loc_map[track_id]
                        xl = (loc[2] - loc[0])/2.0
                        yl = (loc[3] - loc[1])/2.0
                        xn = (x2-x1)/2.0
                        yn = (y2-y1)/2.0
                        speed = int(math.sqrt((xl - xn)**2 + (yl - yn)**2))
                        # print(speed)
                    track_loc_map[track_id] = obj
                
                    color = (0, 255, 0)
                    if speed>5 :
                        color = (0, 0, 255) 
                
                    if track_live.get(track_id) is None : #and class_name !="-" or speed>5:
                        color = (0, 255, 255) 
                        data = {
                            "name": f"{class_name}",
                            "id": track_id,
                            "spd": speed,
                            "box": {"x1": x1, "y1":y1, "x2":x2, "y2": y2},
                            "found":  datetime.now().timestamp()
                        }
                        track_live[track_id] = data
                    else:
                        track_live[track_id]["box"] = {"x1": x1, "y1":y1, "x2":x2, "y2": y2}
                        track_live[track_id]["spd"] = speed
                        

                    # print(f"[{stream_id}] Tracking object {track_id} {x1} {y1} {x2} {y2} {class_name} {speed}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), 10, color, 3)
                    cv2.putText(frame, f"ID: {class_name} {track_id} {speed}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                
                    self.pre_records[stream_id].append(frame)
                    
        
                    if  class_name not in ch_map :
                        ch_map[class_name] = []
                            
                    ch_map[class_name].append(track_live[track_id])
                    
                    
                # print(ch_map)
                
                
                self.channels[stream_id].evaluate(self, stream_id, ch_map)

                    # self.last_frame = frame

                self.display_queues[stream_id].put(frame)

                    
                    
            except Exception as e:
                # Handle the exception
                print(f"Exception Type: {type(e).__name__}")  # Type of exception
                print(f"Exception Message: {str(e)}")        # Exception message
                traceback.print_exc() 

          




    def display_frames(self, stream_id):

        
        # Initialize grid with black frames
        
        
        while True:
            time.sleep(0.01)
            try:
                frame = self.display_queues[stream_id].get()
                while not self.display_queues[stream_id].empty():
                    self.display_queues[stream_id].get()
                
                # last_frames[stream_id] = cv2.resize(frame, (grid_width, grid_height))
                
                self.frames[stream_id] = frame

                row = stream_id // self.grid_cols
                col = stream_id % self.grid_cols
                self.last_frame[row*self.grid_height:(row+1)*self.grid_height, col*self.grid_width:(col+1)*self.grid_width] = frame
                self.display_queues[stream_id].task_done()
                
            except queue.Empty:
                pass    


    # Function to submit jobs from queue to the executor
    def queue_dispatcher(self, executor):
        while True:
            time.sleep(0.001)
            print(f"[{self.task_queue.qsize()}]")
            task = self.task_queue.get()
        
            # executor.submit(task[0], task[1])   
            frame = self.frames[task]
            results = self.ctrl.model(frame, verbose=False)
            frame = results[0].plot()
            self.display_queues[task].put(frame)
            self.task_queue.task_done()    

    def run(self):
        print("Hello, World!")
        motion_threads = []
    

        capture_threads = []
        # Use ThreadPoolExecutor to initialize in parallel
    
        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #     results = executor.map(init_camera, RTSP_SOURCES)

        # for cap in results:
        #     if cap:
        #         caps.append(cap)
        
        for i in range(self.number_of_streams):
            t = threading.Thread(target=self.capture_frames, args=(i,), daemon=True)
            t.start()
            capture_threads.append(t)
            # Create and start live processing threads

        for i in range(self.number_of_streams):
            t = threading.Thread(target=self.motion_detection, args=(i,), daemon=True)
            t.start()
            motion_threads.append(t)


        display_threads = []
        for i in range(self.number_of_streams):
            t = threading.Thread(target=self.display_frames, args=(i,), daemon=True)
            t.start()
            display_threads.append(t)



        # Create and start YOLO processing threads
        executor = ThreadPoolExecutor(max_workers=4)

        dispatcher_thread = threading.Thread(target=self.queue_dispatcher, args=(executor,))
        dispatcher_thread.start()

        yolo_threads = []
        for i in range(self.number_of_streams):
            t = threading.Thread(target=self.object_detection, args=(i,executor,))
            t.start()
            yolo_threads.append(t)


    def run_bk(self):
        

        ctrl = self.ctrl
        
        frame_count = 0 
        skip_frames = 5
        device = "mps"
        width = self.resize_width
        height = self.resize_height
        fps = 15
        
        
        swidth = int(width/2)
        sheight = int(height/2)
        
        
        
        track_ids = {}
        track_live = {}
        track_loc_map ={}
    
        caps = self.caps
       
    
        print(f"caps {len(caps)}")
        
        
        blue_frame = None
        if len(caps)>1:
            blue_frame = np.full((swidth, sheight, 3), (255, 0, 0), dtype=np.uint8)
        else:
            blue_frame = np.full((width, height, 3), (255, 0, 0), dtype=np.uint8)
            
        # Get class names
        class_names = ctrl.model.names
        
        
    
        while(not ctrl.signal.is_set()):
            # print(signal.is_set())
            # print(".")
            # Read a frame from the video
            # ret, frame = cap.read()
            frames = []
            for cap in caps:
                ret, frame = cap.read()
            
                if not ret:
                    print("Warning: Unable to read frame from one of the streams. Using blue frame.")
                    frame = blue_frame  # Use the blue frame if the stream is not ready
                else:
                    if len(caps) >1 :
                        frame = cv2.resize(frame, (swidth  , sheight))  # Resize to fit the grid
                    else:
                        frame = cv2.resize(frame, (width  , height))
                        # frame = cv2.resize(frame, (320  , 320))  # Resize to fit the grid
                frames.append(frame)


            match len(frames):
                case 1:
                    frame = frames[0]
                case 2:
                    top_row = np.hstack(frames[0:2])  # Combine first two frames horizontally
                    frame = np.vstack((top_row))
                    # return "Case 2: Value is 2"
                case 3|4:  # Multiple matches
                    top_row = np.hstack(frames[0:2])  # Combine first two frames horizontally
                    bottom_row = np.hstack(frames[2:4])  # Combine first two frames horizontally
                    frame = np.vstack((top_row, bottom_row))
                    # return "Case 3 or 4: Value is 3 or 4"
                case _:
                    ""

        
            # canvas = np.vstack((top_row, bottom_row))  # Combine the two rows vertically
            frame_count += 1 
            if frame_count % skip_frames != 0:
                continue
        
            canvas = frame

        
            results = ctrl.model(canvas,device=device)

            # Extract detections from the first frame
            # detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class_id]
            res = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class_id]
        
            # detections = []
    #       for r in res:
    #           detections.append(r)

            # Convert to numpy array (SORT requires numpy input)
            detections = np.array(res)
        
            boxes = [det[:5] for det in detections]  # Extract [x1, y1, x2, y2, confidence]
            class_ids = [det[5] for det in detections]  # Extract class IDs

    

            # Create a dictionary to associate track IDs with class IDs
            track_class_map = {}
    
            ch_map = [{},{},{},{}]
    
            try:
            # Update tracker with detections
                tracked_objects = self.tracker.update(detections)
    
                # tracked_objects frmat: [x1, y1, x2, y2, track_id]
                for i in detections:
                    key = f"{int(i[0])}-{int(i[1])}"
                    track_class_map[key] = int(i[5])
            
                
    
                # Draw bounding boxes and track IDs
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = map(int, obj)
            
            
                    class_name = "-"
                    
                    if track_ids.get(track_id) is not None:
                        class_id = track_ids[track_id]
                        class_name = class_names[class_id]
                    else:
                
                        key = f"{int(x1)}-{int(y1)}"
                        if track_class_map.get(key) is not None :
                            class_id = int(track_class_map[key])
                            track_ids[track_id] = class_id
                    
                            class_name = class_names[class_id]
                        
                    speed = 0
                    if track_loc_map.get(track_id) is not None :
                        loc = track_loc_map[track_id]
                        xl = (loc[2] - loc[0])/2.0
                        yl = (loc[3] - loc[1])/2.0
                        xn = (x2-x1)/2.0
                        yn = (y2-y1)/2.0
                        speed = int(math.sqrt((xl - xn)**2 + (yl - yn)**2))
                        # print(speed)
                    track_loc_map[track_id] = obj
                
                    color = (0, 255, 0)
                    if speed>5 :
                        color = (0, 0, 255) 
                
                    if track_live.get(track_id) is None : #and class_name !="-" or speed>5:
                        color = (0, 255, 255) 
                        data = {
                            "name": f"{class_name}",
                            "id": track_id,
                            "spd": speed,
                            "box": {"x1": x1, "y1":y1, "x2":x2, "y2": y2},
                            "found":  datetime.now().timestamp()
                        }
                        track_live[track_id] = data
                    else:
                        track_live[track_id]["box"] = {"x1": x1, "y1":y1, "x2":x2, "y2": y2}
                        track_live[track_id]["spd"] = speed
                        

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {class_name} {track_id} {speed}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                
        
                    ## relocate
                    chid = 0
                    
                    if x1 < swidth :
                        if y1 < sheight :
                            chid = 0 
                        else:
                            chid = 2
                    else:
                        if y1 < sheight :
                            chid = 1
                        else:
                            chid = 3
        
                    if  class_name not in ch_map[chid] :
                        ch_map[chid][class_name] = []
                            
                    ch_map[chid][class_name].append(track_live[track_id])
                    
                    
                # print(ch_map)
                
                self.last_frame = frame
                
                for i, ch in  enumerate(self.channels):
                    
                    ch.evaluate(self, ch_map[i])
                
                    
                
                # if frame is not None : 
                    # cv2.imshow("out", frame)
                
                
            except Exception as e:
                # Handle the exception
                print(f"Exception Type: {type(e).__name__}")  # Type of exception
                print(f"Exception Message: {str(e)}")        # Exception message
                traceback.print_exc() 

          