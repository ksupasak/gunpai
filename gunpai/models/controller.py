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

from models.panel import Panel
from models.channel import Channel
from models.rule import Rule

class Controller:
    def __init__(self, core, number, signal, model, width, height):
        self.num = number
        self.core = core
        self.panels = []
        self.signal = signal
        self.model = model
        self.threads = []
        self.width = width
        self.height = height
        self.mode = 3
        self.fps = 25
        self.pre_record = 10
        self.record_duration = 3
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.class_names = class_names
    def addPanel(self, panel):
        print(f"add Panel")
        
        self.panels.append(panel)
        
    def getPanel(self, i):
        return self.panels[i]
        
    def setFPS(self, fps):
        self.fps = fps
    def setMode(self, mode):
        self.mode = mode
    def setDetectChannel(self, channel):
        self.detect_channel = channel
        
    def run_panel(self, i ):
        panel = self.panels[i]
        panel.run()
        
    def start(self):
     
        
        for i, panel in  enumerate(self.panels):
            p_thread = threading.Thread(target=self.run_panel, args=(i,))
            p_thread.daemon = True
            p_thread.start()
            self.threads.append(p_thread)
            
        for i, t in  enumerate(self.threads):
            t.join()
            
    def ready(self):
        ready = True
        for i, panel in  enumerate(self.panels):
            if panel.ready==False : 
                ready &= False           
        return ready
        