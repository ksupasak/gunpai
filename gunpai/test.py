from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin
import cv2

from models import *

from models.gunpai_yolo import GunpaiYolo
from models.channel import Channel


import subprocess
import paho.mqtt.client as mqtt
import threading
import time



global current_thread
global stop_event
global gunpai
global mode
global detect_channel


FPS = 25

gunpai = None

stop_event = threading.Event()
current_thread = None
# Background Yolo function
def background_yolo(core):
    global current_thread
    global stop_event
    global gunpai
    print(f"Starting background task with data: ")
    # time.sleep(10)  # Simulate a long-running task

    core.start();
    # mqtt_thread = threading.Thread(target=models.yolo.start_yolo)
    # mqtt_thread.daemon = True
    # mqtt_thread.start()

    print("Background task completed!")
    
    
    
def submit(core):
    
    global current_thread
    global stop_event
    global gunpai
      
    if current_thread is not None:
        print("Current Thread exist")
        stop_event.set()
        current_thread.join()
        stop_event = threading.Event()
        current_thread = None
          
    rtmp = "rtmp://localhost:1935/stream/yolo"
        
    rtmp_key = "47yq-9gmc-jq33-e296-8x5s"
        
    # rtmp = f"rtmp://a.rtmp.youtube.com/live2/{rtmp_key}"
    cam = "cam-1"
    # cam = "frigate-1"
    
    
    
    
    
    channels = []
    ids = {}
    
    for i in range(16):
        
        ch = Channel(core, i, f"cam{i}" , cam)
        channels.append(ch)
        ids[ch.name] = ch
        
    
    # ids['cam1'] = "frigate-2"
    
    # Process the form data as needed
    options = {
      "alert": {
        "color": "#ffffff",
        "msg": ""
      },
      "condition": {
        "max_count": "2",
        "min_count": ""
      },
      "panel-1":[
       ids['cam0'],
       ids['cam1'],
       ids['cam2'],
       ids['cam3'],
       ids['cam4'],
       ids['cam5'],
       ids['cam6'],
       ids['cam7'],
       ids['cam8'],
       ids['cam9'],
       ids['cam10'],
       ids['cam11'],
       ids['cam12'],
       ids['cam13'],
       ids['cam14'],
       ids['cam15']
      ] ,
      "detect": "person",
      "event_mq_1": "mqtt://localhost:1883/events/live",
      "event_mq_2": "",
      "model": "yolov8n.pt",
      "rtmp": rtmp,
      "rtmp_key": rtmp_key
    }

    core.config(options,stop_event )
    
    
    current_thread = threading.Thread(target=background_yolo, args=(core,))
    current_thread.daemon = True
    current_thread.start()
    
    

    
    # current_thread.join()
    
    return gunpai
    


core = GunpaiYolo("C001","E00101")

submit(core)

mode = 2
detect_channel = 4



# Loop to continuously capture and display frames
while True:
    # Capture frame-by-frame
    # ret, frame = cap.read()
    
    # If frame is not captured successfully, exit the loop
    # if not ret:
    #     print("Error: Failed to capture frame.")
    #     break
    
    
    
    # Display the captured frame
    time.sleep(1.0/60)
    panel = core.getPanel(0)

    if panel is not None and panel.last_frame is not None :
        
      frame = panel.last_frame

        # position = 
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 1
      color = (0, 0, 255)  # Green
      thickness = 2

      mode_text = "Cap"
      if mode==2 :
          mode_text = "Cap > AI"
      elif mode==3:
          mode_text = "Cap > Motion > AI"
      elif mode==4:
          mode_text = "Cap > Motion"
      
      cv2.putText(frame, mode_text, (50, 100), font, font_scale, color, thickness, cv2.LINE_AA)
      
      fps_text = f"FPS: {FPS}"
      
      cv2.putText(frame, fps_text, (50, 200), font, font_scale, color, thickness, cv2.LINE_AA)

      detect_text = f"Detect: {detect_channel}"
      cv2.putText(frame, detect_text, (50, 300), font, font_scale, color, thickness, cv2.LINE_AA)

    if panel is not None and panel.last_frame is not None :
      cv2.imshow('Gunpai Video Panel-1', panel.last_frame)
   
    
    k = cv2.waitKey(1) 
    

  
    if k & 0xFF == ord('m'):
        mode = (mode+1)
        if mode==5 :
            mode = 1 
    
    if k &0xFF == ord('='):
        FPS = FPS + 1
    elif k &0xFF == ord('-'):
        FPS = FPS - 1

 

    if k &0xFF == ord('2'):
        detect_channel = detect_channel + 1
    elif k &0xFF == ord('1'):
        detect_channel = detect_channel - 1

    SAMPLE_INTERVAL = 1/FPS    


    core.setFPS(FPS)
    core.setDetectChannel(detect_channel)
    core.setMode(mode)
    # print(f"Mode changed to {mode} FPS: {FPS} SAMPLE_INTERVAL: {SAMPLE_INTERVAL}")

    if k & 0xFF == ord('q'):
        break
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break