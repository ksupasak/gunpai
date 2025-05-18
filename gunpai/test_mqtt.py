import paho.mqtt.client as mqtt
import ssl

client = mqtt.Client()
client.username_pw_set("gunpai_user","Minadadmin_")

client.tls_set(
    ca_certs=None,
    certfile=None,
    keyfile=None,
    cert_reqs=ssl.CERT_NONE,
    tls_version=ssl.PROTOCOL_TLSv1_2
)
client.tls_insecure_set(True)


def on_message(client, userdata, msg):
    print(f"📨 Topic: {msg.topic}")
    print(f"📦 Payload: {msg.payload.decode()}")  
client.on_message = on_message

# Called when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to broker")
        client.subscribe("my/topic")  # Replace with your actual topic
    else:
        print(f"❌ Connection failed with code {rc}")


client.on_connect = on_connect

client.connect("mqtt.pcm-life.com", 8883, 60)
client.loop_forever()
