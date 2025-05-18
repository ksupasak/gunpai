import cv2
import torch
# import coremltools as ct
import paho.mqtt.client as mqtt
import math
import numpy as np
from ultralytics import YOLO
from PIL import Image
import subprocess
import traceback
from sort import Sort  # Import the SORT tracking algorithm
import time
from models.queue import Queue

from models.controller import Controller
from models.panel import Panel

from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin
import threading
import requests
import json

from models.models import Controller



class GunpaiYolo:

    _instance = None  # Class attribute to store the singleton instance


    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self, customer_code=None,station_code=None):

        if not hasattr(self, 'initialized'):  # Prevent reinitialization
            self.customer_code = customer_code
            self.station_code = station_code
            self.ctrl = None
            
            self.stream_width = 3840
            self.stream_height = 2160
            
            self.fullframe_width = 3840
            self.fullframe_height = 2160
            
            self.stream_width = 1920
            self.stream_height = 1080

            self.fullframe_width = 1920
            self.fullframe_height = 1080
            
            self.detect_channel = 2
            self.fps = 5
            self.mode = 3        
            self.last_compose = None
         
            self.initialized = True

            print(f"Init Edge {self.station_code}")
          
            self.ctrl_queue =  Queue("mqtt.pcm-life.com", 8883, "gunpai_user","Minadadmin_",f"events/{self.customer_code}/{self.station_code}")
            self.ctrl_queue.start() 
            
            self.event_queue =  Queue("mqtt.pcm-life.com", 8883, "gunpai_user","Minadadmin_",f"events/{self.customer_code}/{self.station_code}")
            self.event_queue.start() 

            self.app_queue =  Queue("mqtt.pcm-life.com", 8883, "gunpai_user","Minadadmin_",f"gunpai/events/{self.customer_code}/{self.station_code}")
            self.app_queue.start() 

            print(f"Init Edge {self.station_code} done")
            print(f"Queue {self.event_queue.mqtt_client.is_connected()}")

    def setFPS(self, fps):
        self.fps = fps
        if self.ctrl is not None :
            self.ctrl.setFPS(fps)
    def setDetectChannel(self, channel):
        self.detect_channel = channel
        if self.ctrl is not None :
            self.ctrl.setDetectChannel(channel)
    def setMode(self, mode):
        self.mode = mode
        if self.ctrl is not None :
            self.ctrl.setMode(mode)

    def getPanel(self, i):
        if self.ctrl is not None and len(self.ctrl.panels)>0:
            return self.ctrl.panels[0]
        return None
            
    def send_message(self, msg):
        self.ctrl_queue.send_message(msg)
    

    def deliver_event(self, event):
        
        
                # Replace with your actual values
        file_path = event['image_path']
        token = 'Y1tpOQF-LasBag9AK_9IxY70nRbFmajQK2aJ-b65'
        account_id = '2525e75bbdede73a8eadba19561b31f8'

        url = f'https://api.cloudflare.com/client/v4/accounts/{account_id}/images/v1'

        headers = {
            'Authorization': f'Bearer {token}'
        }

        files = {
            'file': open(file_path, 'rb')
        }

        response = requests.post(url, headers=headers, files=files)

        # ✅ Check response
        if response.status_code == 200:
            print("Upload success ✅")
            print(response.json())
            event['image_response'] = response.json()
        else:
            print("Upload failed ❌")
            print(response.status_code, response.text)

        #########################################################

        # File and credentials
        file_path = event['video_path']
        
        # Cloudflare Stream API endpoint
        url = f'https://api.cloudflare.com/client/v4/accounts/{account_id}/stream'

        # Headers
        headers = {
            'Authorization': f'Bearer {token}'
        }

        # File payload
        with open(file_path, 'rb') as video_file:
            files = {
                'file': video_file
            }

            response = requests.post(url, headers=headers, files=files)

        # Response handling
        if response.status_code == 200:
            print("✅ Upload successful!")
            print(response.json())
            event['video_response'] = response.json()

        else:
            print("❌ Upload failed:")
            print(response.status_code)
            print(response.text)

        #########################################################


        json_string = json.dumps(event)
        self.event_queue.send_message(json_string)

        
        app_msg = {
            "type": "Alert",
            "detail": f"{event['rule']['name']}",
            "datetime": event['image_response']['result']['uploaded'],
            "video": event['video_response']['result']['playback']['hls'],
            "image": event['image_response']['result']['variants'][0]
        }


        json_string = json.dumps(app_msg)

        self.app_queue.send_message(json_string)

        url = 'https://siaminterlink.com/udirt/www/Api/send_event'
        params = {
            'station': f"{self.customer_code}/{self.station_code}",
            'stream_id': event['stream_id']
        }
        data = {
            "type": "Alert",
            "stream_id": event['stream_id'],
            "detail": f"{event['rule']['name']}",
            "datetime": event['image_response']['result']['uploaded'],
            "video": event['video_response']['result']['playback']['hls'],
            "image": event['image_response']['result']['variants'][0]
        }
        headers = {
            'Content-Type': 'application/json'
        }

        # Serialize and send
        response = requests.post(url, data=json.dumps(data), headers=headers, params=params)
        print("==============================================")
        print('Status Code:', response.status_code)
        print('Response:', response.json())  # or respo
        print("==============================================")

    def detect(self, stream_id, rule, finding, image_path, video_path):
      
        event = {"rule":rule, "finding":finding, "stream_id":stream_id, "image_path":image_path, "video_path":video_path}

        thread = threading.Thread(target=self.deliver_event, args=(event,))
        thread.start()
            
    def version(self):
        return f"{self.station_code} 1.0.0"
        
        
    def ping(self):
        self.ctrl_queue.send_message(f"Ping from {self.version()}")
    
    def alive(self):
        count = 0
        while(True):
            if count%5==0:
                self.ping()
            count+=1    
            time.sleep(1)



    def run_publisher(self, i):
   
        # time.sleep(10)
        print(f"Ready : {self.ctrl.ready()}")
        while(self.ctrl.ready()==False):
            time.sleep(1)
            print("Waiting for Camera ...")
            
        print("Ok for Camera ...")
        while(True):
            try:
            
                time.sleep(1.0/25)
            
                frames = [] 
            
                for panel in self.ctrl.panels : 
                    frame = panel.last_frame
                    # frame = cv2.resize(frame, (int(1920/2)  , int(1080/2)))
                    if isinstance(frame, np.ndarray):
                        frames.append(frame)
                    
                print(f"Frames : {len(frames)}")
                # output_frame = annotated_frame
                if len(frames)==4 : 
                
                    top_row = np.hstack(frames[0:2])  # Combine first two frames horizontally
                    bottom_row = np.hstack(frames[2:4])  # Combine first two frames horizontally
                    aframe = np.vstack((top_row, bottom_row))
                    if self.fullframe_width != self.stream_width : 
                        aframe = cv2.resize(aframe, (self.stream_width  , self.stream_height))
                    
                    print(f"Frames : {len(aframe.tobytes())}")
                    # cv2.imshow('Output Window', aframe)
                    self.last_compose = aframe
                    
                    self.ffmpeg_process.stdin.write(aframe.tobytes())
                   # Press 'q' to exit the video window
                  
                    
            except KeyboardInterrupt:
                print("Streaming stopped.")
            
    def init_panel(self, id, options):
        print(f"initf {options}")
        panel = Panel(id)
        panel.config(options)
        self.panels.append(panel)
        self.ctrl.addPanel(panel)             

    def start(self):
        mqtt_thread = threading.Thread(target=self.alive)
        mqtt_thread.daemon = True
        mqtt_thread.start()

        self.ctrl.start()   
        
    def config(self, options, signal):

        
        print("Start Yolo\n")
        print(f"{options}")
        print("\n")
    
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # default model
        model = YOLO(options["model"]).to(device)
    
        
        self.panels = []
    
        # panels.append(panel)
    
        self.ctrl = Controller(self, 1, signal, model, self.fullframe_width, self.fullframe_height)
        ctrl = self.ctrl
        # threads = []
 #        for i in range(4):
 #            print(f"{i}")
 #            thread = threading.Thread(target=self.init_panel, args=(i,options[f'panel-{i+1}'],))
 #            thread.daemon = True
 #            thread.start()
 #            threads.append(thread)
 #
 #        for i in threads:
 #            i.join
 #     #
 #        print('odfijdif')
        panel = Panel(0)
        panel.config(self, options['panel-1'])
        ctrl.addPanel(panel)
     
        # panel = Panel(1)
        # panel.config(ctrl, options['panel-2'])
        # ctrl.addPanel(panel)

        # panel = Panel(2)
        # panel.config(ctrl, options['panel-3'])
        # ctrl.addPanel(panel)

        # panel = Panel(3)
        # panel.config(ctrl, options['panel-4'])
        # ctrl.addPanel(panel)
    
        # Get class names
        class_names = model.names


        output_rtsp_url = "rtsp://localhost:3554/yolo"
        output_rtmp_url = options["rtmp"]
    
        # blue_frame = np.full((320, 320, 3), (255, 0, 0), dtype=np.uint8)
    
        # cap = caps[0]
        # Get video properties
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    #     frame_count = 0 
    #     skip_frames = 5
    #     device = "mps"
    #     width = self.stream_width
    #     height = self.stream_height
        
    #     bitrate = '4M'
    #     buffersize = '8M'
        
    #     fps = 15

        
    #     ffmpeg_command = [
    #         'ffmpeg',
    #         '-y',  # Overwrite output files without asking
    #         '-f', 'rawvideo',
    #         '-pix_fmt', 'bgr24',
    #         '-s', f"{width}x{height}",
    #         '-r', str(fps),
    #         '-i', '-',  # Input comes from the standard input
    #         '-c:v', 'libx264',
    #         '-preset', 'ultrafast',
    #         '-tune', 'zerolatency',
    #         '-b:v', bitrate,
    #         '-maxrate', bitrate,
    #         '-bufsize', buffersize,
    #         '-vf', f"scale={width}:{height}",
    #         '-pix_fmt', 'yuv420p',
    #         '-f', 'flv',
    #         output_rtmp_url
    #     ]
        
        
    #     self.ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
 

    #     # p_thread = threading.Thread(target=self.run_publisher, args=(0,))
    #     # p_thread.daemon = True
    #     # p_thread.start()
   
    #     ctrl.run(0)
        
        
                

    

    # def start_yolo(self, options, signal):
    #     print("Start Yolo\n")
    #     print(f"{options}")
    #     print("\n")
    
    
    
    
    
    #     tracker = Sort(max_age=100, min_hits=10, iou_threshold=0.5)
    
    #     core = Core()
    #     # Load the YOLOv8 model
    #     # model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # Choose your desired YOLOv8 variant

    #     #model = YOLO("yolov8n-seg.pt")
    #     #model = YOLO("yolov8m.pt")
    #     # model = YOLO("yolov8n-seg.onnx")
    #     # model = YOLO("yolov8n-seg.pt")
    #     # Define the MQTT broker and port
    #     broker = "880b22a0429744488df2940b1696bfe7.s1.eu.hivemq.cloud"  # Replace with your broker address
    #     broker = "880b22a0429744488df2940b1696bfe7.s1.eu.hivemq.cloud"
    
    #     options['event_mq_1'] ="mqtt://localhost:1883/events/live"
    
    
    #     event_mq = options['event_mq_1'].split("/")
    #     event_mq_host = event_mq[2]
    #     event_topic =  event_mq[3:]
    #     event_port = event_mq_host.split(":")[-1]
    #     event_mq_host = event_mq_host.split(":")[0]
    
    #     # topic
    #     # port = 8883                  # Common MQTT port. Use 8883 for SSL.
    #     # topic = "gunpai"         # Replace with your topic
    #     message = "Hello, MQTT!"     # Message to send
    #     #
    #     # send_message("start yolo")
    #     # Callback functions
    #     def on_connect(client, userdata, flags, rc):
    #         if rc == 0:
    #             print("Connected successfully to HiveMQ!")
    #             client.publish(event_topic, message)
    #             print(f"Message '{message}' published to topic '{topic}'")
    #         else:
    #             print(f"Failed to connect, return code {rc}")

    #     def on_disconnect(client, userdata, rc):
    #         print("Disconnected from the broker")
    
    #     # Create an MQTT client instance
    
    #     client = mqtt.Client()
    #     # client.username_pw_set("gunpai", "Minadadmin_2010")
    #     client.on_connect = on_connect
    #     client.on_disconnect = on_disconnect
    

    #     try:
    #         # Connect to the broker
    #         # client.tls_set()  # Optionally specify CA certificates or settings
        
    #         # client.tls_insecure_set(True)
    #         client.connect(event_mq_host, event_port, keepalive=3600)
        

    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    

    
    
    #     objects = options["detect"].split(",")
    #     max_count = None
    #     min_count = None
    
    #     max_count = int(options["condition"]["max_count"])

    #     # model = YOLO("yolov8s.pt")
    
    #     model = YOLO(options["model"])
    
    #     # Get class names
    #     class_names = model.names

    #     output_rtsp_url = "rtsp://localhost:3554/yolo"
    #     # output_rtmp_url = "rtmp://localhost:1935/stream/yolo"
    
    #     output_rtmp_url = options["rtmp"]
    

    #     # YOLO hardware process

    #     if torch.backends.mps.is_available():
    #         device = torch.device("mps")
    #     else:
    #         device = torch.device("cpu")
    #     # Enable GPU acceleration if available
    #     if torch.cuda.is_available():
    #         model.cuda() 
    
    #     device = "mps"
    
    

    #     skip_frames = 1  # Process every 5th frame

    #     frame_count = 0  # Initialize frame count
    
    
    
    
    
    #     rtsp_url = 'rtsp://127.0.0.1:8554/cam1'


    #     # Define your RTSP stream URLs
    #     rtsp_urls = [
    #         'rtsp://127.0.0.1:8554/cam1',
    #         'rtsp://127.0.0.1:8554/cam2',
    #         'rtsp://127.0.0.1:8554/cam3',
    #         'rtsp://127.0.0.1:8554/cam4',
    #     ]
    
    #     caps = []
    
    #     if options['panel1'] :
    #         for i, s in  enumerate(options['panel1']):
    #             print(f"{i} {s} {s is not None}")
    #             if s is not None:
    #                 token = s.split("-")
    #                 type = token[0]
    #                 channel = token[1]
    #                 print(f"{type} {channel}")
    #                 if type=="frigate":
    #                     rtsp_url = rtsp_urls[int(channel)-1]
    #                     cap = cv2.VideoCapture(rtsp_url)
    #                     if cap.isOpened():
    #                         caps.append(cap)
    #                 if type=="cam":
    #                     cap = cv2.VideoCapture(int(channel)-1)
    #                     if cap.isOpened():
    #                         caps.append(cap)
                
    
    
    #     # caps = [cv2.VideoCapture(rtsp_url) for rtsp_url in rtsp_urls]
    # # Check if all streams are opened successfully
    #     for i, cap in enumerate(caps):
    #         if not cap.isOpened():
    #             print(f"Error: Unable to open the RTSP stream {i + 1}.")
    #             exit()
    #     # rtsp_url = 'rtsp://127.0.0.1:8554/cam1'
    #     # cap = cv2.VideoCapture(rtsp_url)
    #     # Open the video capture
    #     # cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify video file path
   
   
    #     # option2  
    #     #cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/bodycam")
    
    
    #     # blue_frame = np.full((320, 320, 3), (255, 0, 0), dtype=np.uint8)
    #     blue_frame = np.full((1280, 720, 3), (255, 0, 0), dtype=np.uint8)
    
    #     # cap = caps[0]
    #     # Get video properties
    #     # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     # fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    #     # width = 1280
    #     # height = 720
    #     # fps = 15
    
    #     width = 1920
    #     height = 1080
    #     fps = 15
    

    #     # FFmpeg command to stream to RTSP
    #     ffmpeg_command = [
    #         'ffmpeg',
    #         '-y',  # Overwrite output files without asking
    #         '-f', 'rawvideo',
    #         '-vcodec', 'rawvideo',
    #         '-pix_fmt', 'bgr24',
    #         '-s', f"{width}x{height}",
    #         '-r', str(15),
    #         '-i', '-',  # Input comes from the standard input
    #         '-c:v', 'libx264',
    #         '-preset', 'ultrafast',
    #         '-f', 'rtsp',
    #         '-rtsp_transport','tcp',
    #         output_rtsp_url
    #     ]
    #     ffmpeg_command = [
    #         'ffmpeg',
    #         '-y',  # Overwrite output files without asking
    #         '-f', 'rawvideo',
    #         '-vcodec', 'rawvideo',
    #         '-pix_fmt', 'bgr24',
    #         '-s', f"{width}x{height}",
    #         '-r', str(15),
    #         '-i', '-',  # Input comes from the standard input
    #         '-c:v', 'libx264',
    #         '-preset', 'ultrafast',
    #         '-pix_fmt', 'yuv420p',
    #         '-f', 'flv',
    #         output_rtmp_url
    #     ]
    
    #     ffmpeg_command = [
    #         'ffmpeg',
    #         '-y',  # Overwrite output files without asking
    #         '-f', 'rawvideo',
    #         '-vcodec', 'rawvideo',
    #         '-pix_fmt', 'bgr24',
    #         '-s', f"{width}x{height}",
    #         '-r', str(30),
    #         '-i', '-',  # Input comes from the standard input
    #         '-vcodec', 'libx264',
    #         '-preset', 'ultrafast',
    #         '-tune', 'zerolatency',
    #         '-b:v', '4M',
    #         '-maxrate', '4M',
    #         '-bufsize', '8M',
    #         '-vf', 'scale=1920:1080',
    #         '-acodec', 'aac',
    #         '-b:a', '128k',
    #         '-f', 'flv',
    #         output_rtmp_url
    #     ]
    #     ffmpeg_command = [
    #         'ffmpeg',
    #         '-y',  # Overwrite output files without asking
    #         '-f', 'rawvideo',
    #         '-pix_fmt', 'bgr24',
    #         '-s', f"{width}x{height}",
    #         '-r', str(15),
    #         '-i', '-',  # Input comes from the standard input
    #         '-c:v', 'libx264',
    #         '-preset', 'ultrafast',
    #         '-tune', 'zerolatency',
    #         '-b:v', '4M',
    #         '-maxrate', '4M',
    #         '-bufsize', '8M',
    #         '-vf', 'scale=1920:1080',
    #         '-pix_fmt', 'yuv420p',
    #         '-f', 'flv',
    #         output_rtmp_url
    #     ]
    #     ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    #     track_ids = {}
    #     track_live = {}
    #     track_loc_map ={}
    
    
    #     print(f"caps {len(caps)}")
    
    #     core.send_message('Ready to Start')
    
    #     while(not signal.is_set()):
    #         # print(signal.is_set())
    #         # print(".")
    #         # Read a frame from the video
    #         # ret, frame = cap.read()
    #         frames = []
    #         for cap in caps:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 print("Warning: Unable to read frame from one of the streams. Using blue frame.")
    #                 frame = blue_frame  # Use the blue frame if the stream is not ready
    #             else:
    #                 if len(caps) >1 :
    #                     frame = cv2.resize(frame, (960  , 540))  # Resize to fit the grid
    #                     # frame = cv2.resize(frame, (320  , 320))  # Resize to fit the grid
                    
    #                 ""

    #             frames.append(frame)
    #         # print(f". {len(frames)}")    
    #         # ret, frame = caps[0].read()
    #         # cap = caps[0]
    #         # ret, frame = cap.read()
        
        
    #         # frames.append(frame)
    #         # frames.append(frame)
    #         # frames.append(frame)
    #         # frames.append(frame)
    #         #
    #         #
    #         # # Combine the frames into a single canvas (2x2 grid)
    #         # top_row = np.hstack(frames[0:2])  # Combine first two frames horizontally
    #         # bottom_row = np.hstack(frames[2:4])  # Combine last two frames horizontally
    #         # # top_row_2 = np.hstack(frames[8:12])  # Combine first two frames horizontally
    #         # # bottom_row_2 = np.hstack(frames[12:16])  # Combine last two frames horizontally
    #         #
 
    #         match len(frames):
    #             case 1:
    #                 frame = frames[0]
    #             case 2:
    #                 top_row = np.hstack(frames[0:2])  # Combine first two frames horizontally
    #                 frame = np.vstack((top_row))
    #                 # return "Case 2: Value is 2"
    #             case 3|4:  # Multiple matches
    #                 top_row = np.hstack(frames[0:2])  # Combine first two frames horizontally
    #                 bottom_row = np.hstack(frames[2:4])  # Combine first two frames horizontally
    #                 frame = np.vstack((top_row, bottom_row))
    #                 # return "Case 3 or 4: Value is 3 or 4"
    #             case _:
    #                 # return "Default case: Value not matched"
    #                 ""
        
    #         # print(f". xxx {len(frames)}")    
        
        
    #         # canvas = np.vstack((top_row, bottom_row))  # Combine the two rows vertically
    #         frame_count += 1 
    #         if frame_count % skip_frames != 0:
    #             continue
        
    #         canvas = frame
        
    #         # for core ml
        
    #         # # Preprocess the frame (resize and normalize)
    #         # resized_frame = cv2.resize(canvas, (640, 640))  # Resize to match model input shape
    #         # normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    #         #
    #         # # Convert to Core ML compatible format (CVPixelBuffer)
    #         # # image_input = ct.ImageType(name="image", shape=(1, 3, 640, 640))
    #         #
    #         # # Convert the NumPy array to a CVPixelBuffer
    #         # # coreml_input = image_input.from_numpy_array(normalized_frame)
    #         # pil_image = Image.fromarray(np.uint8(normalized_frame * 255))  # Convert to PIL Image
    
    #         # coreml_input = ct.pixel_buffer_from_numpy_array(normalized_frame)
    
    #         # try:
    #    #          output = model.predict({'image': pil_image})
    #    #      except Exception as e:
    #    #          print(f"Error during prediction: {e}")
    #    #          continue
    #    #
    #         results = model(canvas,device=device)


    #         # annotated_frame = results[0].plot()
        
        
    #         # Extract detections from the first frame
    #         # detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class_id]
    #         res = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class_id]
        
    #         # detections = []
    #   #       for r in res:
    #   #           detections.append(r)

    #         # Convert to numpy array (SORT requires numpy input)
    #         detections = np.array(res)
        
    #         boxes = [det[:5] for det in detections]  # Extract [x1, y1, x2, y2, confidence]
    #         class_ids = [det[5] for det in detections]  # Extract class IDs

      

    #         # Create a dictionary to associate track IDs with class IDs
    #         track_class_map = {}
       
        
    
    #         try:
    #         # Update tracker with detections
    #             tracked_objects = tracker.update(detections)
      
    #             # tracked_objects frmat: [x1, y1, x2, y2, track_id]
    #             for i in detections:
    #                 key = f"{int(i[0])}-{int(i[1])}"
    #                 track_class_map[key] = int(i[5])
             
      
      
    #             # Draw bounding boxes and track IDs
    #             for obj in tracked_objects:
    #                 x1, y1, x2, y2, track_id = map(int, obj)
               
              
    #                 class_name = "-"
    #                 if track_ids.get(track_id) is not None:
    #                     class_id = track_ids[track_id]
    #                     class_name = class_names[class_id]
    #                 else:
                  
    #                     key = f"{int(x1)}-{int(y1)}"
    #                     if track_class_map.get(key) is not None :
    #                         class_id = int(track_class_map[key])
    #                         track_ids[track_id] = class_id
                       
    #                         class_name = class_names[class_id]
                        
    #                 speed = 0
    #                 if track_loc_map.get(track_id) is not None :
    #                     loc = track_loc_map[track_id]
    #                     xl = (loc[2] - loc[0])/2.0
    #                     yl = (loc[3] - loc[1])/2.0
    #                     xn = (x2-x1)/2.0
    #                     yn = (y2-y1)/2.0
    #                     speed = int(math.sqrt((xl - xn)**2 + (yl - yn)**2))
    #                     # print(speed)
    #                 track_loc_map[track_id] = obj
                
    #                 color = (0, 255, 0)
    #                 if speed>5 :
    #                     color = (0, 0, 255)  
    #                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #                 cv2.putText(frame, f"ID: {class_name} {track_id} {speed}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
    #                 if track_live.get(track_id) is None and class_name !="-" or speed>5:  
                                 
    #                     data = {
    #                         "name": f"{class_name}",
    #                         "id": track_id,
    #                         "spd": speed,
    #                         "box": {"x1": x1, "y1":y1, "x2":x2, "y2": y2},
    #                         "live": 30
    #                     }
                    
    #                     track_live[track_id] = data
                    
    #                     core.detect(data)
                
    #         except Exception as e:
    #             print(f"Caught an error: {e}")
    #             traceback.print_exc() 
            
    #         # detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class_id]

    #         # # Load the original frame (e.g., from canvas)
    #         # frame = np.array(canvas)  # Ensure this is a numpy array (YOLO models use numpy or PIL images)
    #         # found = 0
    #         # # Iterate through detections and draw bounding boxes
    #         # for detection in detections:
    #         #     x1, y1, x2, y2, confidence, class_id = detection
    #         #
    #         #     # Convert coordinates to integers
    #         #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         #
    #         #     class_name = class_names[class_id]  # Get the class name
    #         #
    #         #
    #         #     if class_name in objects :
    #         #         print(f"Found : {class_name}")
    #         #         found +=1
    #         #
    #         #
    #         #     # Define label and color
    #         #     label = f"Class {class_name}: {confidence:.2f}"
    #         #     color = (0, 255, 0)  # Green for bounding boxes
    #         #
    #         #     # Draw the bounding box
    #         #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
    #         #
    #         #     # Draw the label
    #         #     text_size, _ = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
    #         #     text_w, text_h = text_size
    #         #     cv2.rectangle(frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, thickness=-1)  # Filled rectangle for label
    #         #     cv2.putText(frame, label, (x1, y1 - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
    #         #
    #         # if False and max_count is not None and found >= max_count :
    #         #     print(f"Max found {found}")
    #         #     try:
    #         #         # Connect to the broker
    #         #         # client.connect(broker, port, keepalive=60)
    #         #         message = f"Message : found '{objects}' = {found} to {topic}"
    #         #         # Publish the message
    #         #         res = client.publish(topic, message)
    #         #         print(f"{message} {res}")
    #         #
    #         #
    #         #
    #         #         # Disconnect from the broker
    #         #
    #         #
    #         #     except Exception as e:
    #         #         print(f"An error occurred: {e}")
    #         #
    #         annotated_frame=frame
    #         # annotated_frame= canvas
        
    #     #
    #     #
    #     #     output_frame = cv2.resize(annotated_frame, (1920  , 1080))
        
    #         annotated_frame = cv2.resize(annotated_frame, (1920  , 1080))
        
    #         output_frame = annotated_frame
        
        
        
    #         # print(f"size = {len(output_frame.tobytes())}")
    
    #         try:
    #             ffmpeg_process.stdin.write(output_frame.tobytes())
    #         except KeyboardInterrupt:
    #             print("Streaming stopped.")
     
        
     
    
   
    #         # Visualize the results on the frame
    

    #         # Display the annotated frame
    #         # cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    #         # Break the loop if 'q' is pressed
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
            
            
    #     print("Subprocess terminated.")
    #     # Release the video capture and destroy all windows
    #     cap.release()
    #     ffmpeg_process.kill()
    #     ffmpeg_process.wait()
    #     cv2.destroyAllWindows()
    #     print("Subprocess terminated.")
