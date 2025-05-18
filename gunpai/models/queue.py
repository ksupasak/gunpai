import paho.mqtt.client as mqtt
import threading
import time
import json

import ssl

class Queue:
    
    def __init__(self, broker, port, username, password, topic):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.mqtt_client = None
        self.mqtt_thread = None
        self.mqtt_topic = topic
        
    # MQTT client functions
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT broker with result code {rc}")
        client.subscribe( self.mqtt_topic)

    def on_message(self, client, userdata, msg):
        print(f"Received message: {msg.payload.decode()} on topic {msg.topic}")
        

    def send_message(self, message, request_topic = "default" ):
        # mqtt_client = mqtt.Client()
        # mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        topic = self.mqtt_topic
        if request_topic!= "default" :
            topic = request_topic
        self.mqtt_client.publish(topic, message)
        # mqtt_client.disconnect()
    def on_log(self, client, userdata, level, buf):
        print(f"📋 Log: {buf}")

    # Background MQTT thread
    def start_mqtt(self):
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(self.username, self.password)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_log = self.on_log 

        self.mqtt_client.tls_set(
            ca_certs=None,
            certfile=None,
            keyfile=None,
            cert_reqs=ssl.CERT_NONE,
            tls_version=ssl.PROTOCOL_TLSv1_2
        )
        self.mqtt_client.tls_insecure_set(True)


        self.mqtt_client.connect(self.broker, self.port, 60)

        self.mqtt_client.loop_forever()
        
    def start(self):
        mqtt_thread = threading.Thread(target=self.start_mqtt)
        mqtt_thread.daemon = True
        mqtt_thread.start()
