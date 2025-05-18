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
import json

class Rule:
    
    def __init__(self,core, id, name, obj, nobj, count, duration):
        self.core = core
        self.id = id
        self.name = name
        self.obj = obj
        self.nobj = nobj
        self.count = count
        
        self.duration = duration
        self.found = None
        self.active = True
        self.alert_at = None
        self.delay = 10
        self.ids = []
    def dump(self):
        data = {
            'name': self.name,
            'obj': self.obj,
            'nobj': self.nobj,
            'count': self.count,
            'duration': self.duration,
            'delay': self.delay,
            'id': self.id,
            'active': self.active,
            'alert_at': self.alert_at,
            'found': self.found
        }
        return data
    
    def evaluate(self, panel, stream_id, scene):
        
        # s = scene.get(self.obj)
        ox = 100
        oy = 400
        s = []
        
        if any(key in scene for key in self.obj): #s is not None :
        
            s = sum((scene[k] for k in self.obj if k in scene), [])

            if self.nobj is None or scene.get(self.nobj) is None :
                print(f"nobj {self.nobj}")
                print(scene)
                print(f'check -1  {self.count} {len(s)}')
                if self.count <= len(s) :
                    print('check 0')
                    if self.found is None :
                        print('check 1')
                        self.found = datetime.now().timestamp()
                    print(f'check 1.5 {self.found}')
                    print(f'dx {datetime.now().timestamp() - self.found} {self.duration}')
                    if datetime.now().timestamp() - self.found > self.duration :
                        print(f'check 2 {self.alert_at}')
                        if self.alert_at is None :  

                           
                            
                            image_path = panel.write_capture(stream_id)
                            video_path = panel.write_pre_record(stream_id)
                            rule_serial = self.dump()

                            print(f"rule_serial {rule_serial}")
                            self.core.detect(stream_id, rule_serial, scene, image_path, video_path)

                            self.alert_at = datetime.now().timestamp()
                            print(f"xxAlert {scene}")
                        else:
                            if datetime.now().timestamp() - self.alert_at > self.delay :
                                self.alert_at = None
                                self.found = None
                        # cv2.rectangle(panel.last_frame, (ox+20, oy+20), (int(ox + datetime.now().timestamp() - self.found), oy+40),  (255, 0, 0) , 2)
                        if self.found is not None:
                            cv2.rectangle(panel.last_frame, (ox+10, oy+10+self.id*40), (ox+int(10*(datetime.now().timestamp() - self.found))+10, oy+30+self.id*40), (255, 0, 0) , -1)
                            cv2.putText(panel.last_frame, f"ID: {self.name} ", (ox+10, oy+20+self.id*40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                             
                    return True
                    
        self.found = None
        return False   
       
       
       
       