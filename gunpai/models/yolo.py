import cv2
import torch
import coremltools as ct
import paho.mqtt.client as mqtt
import math
import numpy as np
from ultralytics import YOLO
from PIL import Image
import subprocess
import traceback
from sort import Sort  # Import the SORT tracking algorithm


def start_yolo(options, signal):
    print("Start Yolo\n")
    print(f"{options}")
    print("\n")
    
    tracker = Sort(max_age=100, min_hits=10, iou_threshold=0.5)
    

    # Load the YOLOv8 model
    # model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # Choose your desired YOLOv8 variant

    #model = YOLO("yolov8n-seg.pt")
    #model = YOLO("yolov8m.pt")
    # model = YOLO("yolov8n-seg.onnx")
    # model = YOLO("yolov8n-seg.pt")
    # Define the MQTT broker and port
    broker = "880b22a0429744488df2940b1696bfe7.s1.eu.hivemq.cloud"  # Replace with your broker address
    broker = "880b22a0429744488df2940b1696bfe7.s1.eu.hivemq.cloud"
    port = 8883                  # Common MQTT port. Use 8883 for SSL.
    topic = "gunpai"         # Replace with your topic
    message = "Hello, MQTT!"     # Message to send
    
    
    # Callback functions
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected successfully to HiveMQ!")
            client.publish(topic, message)
            print(f"Message '{message}' published to topic '{topic}'")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_disconnect(client, userdata, rc):
        print("Disconnected from the broker")
    
    # Create an MQTT client instance
    
    client = mqtt.Client()
    client.username_pw_set("gunpai", "Minadadmin_2010")
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    

    try:
        # Connect to the broker
        # client.tls_set()  # Optionally specify CA certificates or settings
        
        # client.tls_insecure_set(True)
        client.connect(broker, port, keepalive=3600)
        

    except Exception as e:
        print(f"An error occurred: {e}")
    
    
    objects = options["detect"].split(",")
    max_count = None
    min_count = None
    
    max_count = int(options["condition"]["max_count"])

    # model = YOLO("yolov8s.pt")
    
    model = YOLO(options["model"])
    
    # Get class names
    class_names = model.names

    output_rtsp_url = "rtsp://localhost:3554/yolo"
    # output_rtmp_url = "rtmp://localhost:1935/stream/yolo"
    
    output_rtmp_url = options["rtmp"]
    

    # YOLO hardware process

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # Enable GPU acceleration if available
    if torch.cuda.is_available():
        model.cuda() 
    
    device = "mps"
    
    

    skip_frames = 1  # Process every 5th frame

    frame_count = 0  # Initialize frame count
    
    
    
    
    
    rtsp_url = 'rtsp://127.0.0.1:8554/cam1'


    # Define your RTSP stream URLs
    rtsp_urls = [
        'rtsp://127.0.0.1:8554/cam1',
        'rtsp://127.0.0.1:8554/cam2',
        'rtsp://127.0.0.1:8554/cam3',
        'rtsp://127.0.0.1:8554/cam4',
    ]
    
    caps = []
    
    if options['panel1'] :
        for i, s in  enumerate(options['panel1']):
            print(f"{i} {s} {s is not None}")
            if s is not None:
                token = s.split("-")
                type = token[0]
                channel = token[1]
                print(f"{type} {channel}")
                if type=="frigate":
                    rtsp_url = rtsp_urls[int(channel)-1]
                    cap = cv2.VideoCapture(rtsp_url)
                    if cap.isOpened():
                        caps.append(cap)
                if type=="cam":
                    cap = cv2.VideoCapture(int(channel)-1)
                    if cap.isOpened():
                        caps.append(cap)
                
    
    
    # caps = [cv2.VideoCapture(rtsp_url) for rtsp_url in rtsp_urls]
# Check if all streams are opened successfully
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Unable to open the RTSP stream {i + 1}.")
            exit()
    # rtsp_url = 'rtsp://127.0.0.1:8554/cam1'
    # cap = cv2.VideoCapture(rtsp_url)
    # Open the video capture
    # cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify video file path
   
   
    # option2  
    #cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/bodycam")
    
    
    # blue_frame = np.full((320, 320, 3), (255, 0, 0), dtype=np.uint8)
    blue_frame = np.full((1280, 720, 3), (255, 0, 0), dtype=np.uint8)
    
    # cap = caps[0]
    # Get video properties
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # width = 1280
    # height = 720
    # fps = 15
    
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
    
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{width}x{height}",
        '-r', str(30),
        '-i', '-',  # Input comes from the standard input
        '-vcodec', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-b:v', '4M',
        '-maxrate', '4M',
        '-bufsize', '8M',
        '-vf', 'scale=1920:1080',
        '-acodec', 'aac',
        '-b:a', '128k',
        '-f', 'flv',
        output_rtmp_url
    ]
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{width}x{height}",
        '-r', str(15),
        '-i', '-',  # Input comes from the standard input
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-b:v', '4M',
        '-maxrate', '4M',
        '-bufsize', '8M',
        '-vf', 'scale=1920:1080',
        '-pix_fmt', 'yuv420p',
        '-f', 'flv',
        output_rtmp_url
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    track_ids = {}
    track_loc_map ={}
    
    
    print(f"caps {len(caps)}")
    while(not signal.is_set()):
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
                    frame = cv2.resize(frame, (960  , 540))  # Resize to fit the grid
                    # frame = cv2.resize(frame, (320  , 320))  # Resize to fit the grid
                    
                ""

            frames.append(frame)
        # print(f". {len(frames)}")    
        # ret, frame = caps[0].read()
        # cap = caps[0]
        # ret, frame = cap.read()
        
        
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
                # return "Default case: Value not matched"
                ""
        
        # print(f". xxx {len(frames)}")    
        
        
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


        # annotated_frame = results[0].plot()
        
        
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
       
        
    
        try:
        # Update tracker with detections
            tracked_objects = tracker.update(detections)
      
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {class_name} {track_id} {speed}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        except Exception as e:
            print(f"Caught an error: {e}")
            traceback.print_exc() 
            
        # detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class_id]

        # # Load the original frame (e.g., from canvas)
        # frame = np.array(canvas)  # Ensure this is a numpy array (YOLO models use numpy or PIL images)
        # found = 0
        # # Iterate through detections and draw bounding boxes
        # for detection in detections:
        #     x1, y1, x2, y2, confidence, class_id = detection
        #
        #     # Convert coordinates to integers
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #
        #     class_name = class_names[class_id]  # Get the class name
        #
        #
        #     if class_name in objects :
        #         print(f"Found : {class_name}")
        #         found +=1
        #
        #
        #     # Define label and color
        #     label = f"Class {class_name}: {confidence:.2f}"
        #     color = (0, 255, 0)  # Green for bounding boxes
        #
        #     # Draw the bounding box
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        #
        #     # Draw the label
        #     text_size, _ = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        #     text_w, text_h = text_size
        #     cv2.rectangle(frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, thickness=-1)  # Filled rectangle for label
        #     cv2.putText(frame, label, (x1, y1 - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
        #
        # if False and max_count is not None and found >= max_count :
        #     print(f"Max found {found}")
        #     try:
        #         # Connect to the broker
        #         # client.connect(broker, port, keepalive=60)
        #         message = f"Message : found '{objects}' = {found} to {topic}"
        #         # Publish the message
        #         res = client.publish(topic, message)
        #         print(f"{message} {res}")
        #
        #
        #
        #         # Disconnect from the broker
        #
        #
        #     except Exception as e:
        #         print(f"An error occurred: {e}")
        #
        annotated_frame=frame
        # annotated_frame= canvas
        
    #
    #
    #     output_frame = cv2.resize(annotated_frame, (1920  , 1080))
        
        annotated_frame = cv2.resize(annotated_frame, (1920  , 1080))
        
        output_frame = annotated_frame
        
        
        
        # print(f"size = {len(output_frame.tobytes())}")
    
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
    print("Subprocess terminated.")
    # Release the video capture and destroy all windows
    cap.release()
    ffmpeg_process.kill()
    ffmpeg_process.wait()
    cv2.destroyAllWindows()
    print("Subprocess terminated.")
