import cv2
from sort import Sort  # Import the SORT tracking algorithm
import numpy as np
import threading
import time
import math
import traceback
import queue
import json
from math import sqrt

from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from models.rule import Rule
                     

                        
       
class Channel(json.JSONEncoder):
   def __init__(self,core, id, name, source):
       self.id = id
       self.name = name
       self.source = source
      
       self.last_frame = None
       self.rules = []
       
       
       # self.rules.append(Rule(0, "person", "person", None, 1, 1))
       # self.rules.append(Rule(1, "left backpack", ["bottle","cup"], "person", 1, 3))
    #    self.rules.append(Rule(1, "left backpack", ["backpack","suitcase"], "person", 1, 3))
       self.rules.append(Rule(core, 1, "Cell Phone Detection", ["cell phone"], None , 1, 3))
        
     
            
   def open(self, panel, ox, oy):
       self.panel = panel
       self.ox = ox
       self.oy = oy
        
       rtsp_urls = [
           'rtsp://127.0.0.1:8554/cam1',
           'rtsp://127.0.0.1:8554/cam2',
           'rtsp://127.0.0.1:8554/cam3',
           'rtsp://127.0.0.1:8554/cam4',
           'rtsp://127.0.0.1:8554/cam5',
           'rtsp://10.149.1.62:8554/stream/test',
       ]
       if self.source is not None:
           
           token = self.source.split(":")
           type = token[0]
           channel = token[1]
           
           print(f"{type} {channel}")
           capture_id = None
           if type=="frigate":
               rtsp_url = rtsp_urls[int(channel)-1]
               capture_id = rtsp_url 
              
           if type=="cam":
               capture_id = int(channel)-1
           if type=="rtsp":
               capture_id = self.source
           if type=="https":
               capture_id = self.source
               
       print(f"capture_id {capture_id}")
       self.capture_id = capture_id
       self.cap =  cv2.VideoCapture(capture_id)
       self.ready = False
       if self.cap.isOpened():
           self.ready = True
           return self.cap
       else:
           return None
    


   def evaluate(self, panel, stream_id, scene):
       
       for r in self.rules :
           
           r.evaluate(panel, stream_id, scene)
   

           