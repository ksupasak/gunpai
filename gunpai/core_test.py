from models.core import Core
import threading
import time

core = Core("C0001","edge-00000")

core.start()

count = 0 
while(True):
    time.sleep(1)
    print(".")