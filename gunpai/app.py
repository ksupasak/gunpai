from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin

import models.yolo
import models.yolo2x2



import subprocess
import paho.mqtt.client as mqtt
import threading
import time

app = Flask(__name__)

# Create Blueprint
auth = Blueprint("auth", __name__)

app.secret_key = "minadadmin"  # Change this in production

# Register the auth blueprint
app.register_blueprint(auth, url_prefix="/auth")

# In-Memory User Store (Replace with a database for production)
users = {"admin": {"password": "kamsk"}}


# Define a flag for stopping the thread
stop_event = threading.Event()

current_thread = None

# User Class for Flask-Login
class User(UserMixin):
    def __init__(self, username):
        self.id = username



# Configure Flask-Login
login_manager = LoginManager()
login_manager.login_view = "login"  # Redirect to login if not authenticated
login_manager.init_app(app)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(username):
    if username in {"admin"}:  # Replace with a database query in production
        return User(username)
    return None


# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "events"

# Global variable for background MQTT messages
mqtt_messages = []


# MQTT client functions
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload.decode()} on topic {msg.topic}")
    mqtt_messages.append({
        "topic": msg.topic,
        "message": msg.payload.decode()
    })
    if(msg.payload.decode()=="restart"):
        restart_frigate()

# Background MQTT thread
def start_mqtt():
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_forever()

# Flask routes
@app.route('/publish', methods=['POST'])
def publish_message():
    data = request.json
    topic = data.get("topic", MQTT_TOPIC)
    message = data.get("message", "")
    
    mqtt_client = mqtt.Client()
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.publish(topic, message)
    mqtt_client.disconnect()
    
    return jsonify({"status": "Message published", "topic": topic, "message": message})

# Controller
@app.route('/')
@login_required
def dashboard():
    return render_template("dashboard.html", title="Dashboard")

# User loader for Flask-Login
@login_manager.user_loader
def load_user(username):
    if username in {"admin"}:  # Replace with a database query in production
        return User(username)
    return None    

@app.route("/usersx")
@login_required
def usersx():
    return render_template("users.html", title="Users")

@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html", title="Settings") 

@app.route('/messages', methods=['GET'])
def get_messages():
    return jsonify(mqtt_messages)



@app.route('/restart-frigate', methods=['GET'])
def restart_frigate():
    try:
        subprocess.run(["docker", "restart", "frigate"], check=True)
        return {"message": "Frigate restarted successfully"}, 200
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}, 500

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users and users[username]["password"] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials. Please try again.", "danger")
    return render_template("login.html", title="Users")#render_template("login.html", title="Login")


# Logout route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# Background Yolo function
def background_yolo( signal, options):
    print(f"Starting background task with data: ")
    # time.sleep(10)  # Simulate a long-running task

    models.yolo.start_yolo(options, signal)

    # mqtt_thread = threading.Thread(target=models.yolo.start_yolo)
    # mqtt_thread.daemon = True
    # mqtt_thread.start()

    print("Background task completed!")
    
    
    
    

@app.route('/start-task', methods=['GET'])
def start_task():
    # data = request.json.get('data')
    thread = threading.Thread(target=background_yolo, args=({}))
    thread.daemon = True
    thread.start()
    
    return "This is plain text", 200, {'Content-Type': 'text/plain'}


# Background function
def yolo2x2_task():
    print(f"Starting background task with data: ")
    # time.sleep(10)  # Simulate a long-running task

    mqtt_thread = threading.Thread(target=models.yolo2x2.start_yolo)
    mqtt_thread.daemon = True
    mqtt_thread.start()

    print("Background task completed!")

@app.route('/start-yolo2x2', methods=['GET'])
def start_yolo2x2():
    # data = request.json.get('data')
    thread = threading.Thread(target=yolo2x2_task, args=())
    thread.start()
    return "This is plain text", 200, {'Content-Type': 'text/plain'}

@app.route('/detector')
def index():
    
# Example array of key-value pairs
    options = [
        {"key": "cam-1", "value": "Camera 1"},
        {"key": "cam-2", "value": "Camera 2"},
        {"key": "frigate-1", "value": "Frigate 1"},
        {"key": "frigate-2", "value": "Frigate 2"},
        {"key": "frigate-3", "value": "Frigate 3"},
        {"key": "frigate-4", "value": "Frigate 4"}
        
    ]
    models = [
        {"key": "yolov8n.pt", "value": "yolov8n.pt"},
        {"key": "yolov11s.pt", "value": "yolov11s.p"},
        {"key": "yolov8x.pt", "value": "yolov8x.pt"},
        {"key": "yolov8n-seg.pt", "value": "yolov8n-seg.pt"},
    ]
    return render_template('detector.html',options=options, models=models)

@app.route('/detector_submit', methods=['POST'])
def submit():
    
    global current_thread
    global stop_event
      
    if current_thread is not None:
        print("Current Thread exist")
        stop_event.set()
        current_thread.join()
        stop_event = threading.Event()
        current_thread = None
    
    data = request.form
    # Process the form data as needed
    response = {
        "panel1": [data.get('panel1_1'), data.get('panel1_2'), data.get('panel1_3'), data.get('panel1_4')],
        "panel2": [data.get('panel2_1'), data.get('panel2_2'), data.get('panel2_3'), data.get('panel2_4')],
        "detect": data.get('detect'),
        "model": data.get('model'),
        "rtmp": data.get('rtmp'),
        "condition": {
            "min_count": data.get('min_count'),
            "max_count": data.get('max_count')
        },
        "alert": {
            "msg": data.get('msg'),
            "color": data.get('color')
        }
    }
    
    current_thread = threading.Thread(target=background_yolo, args=(stop_event,response,))
    current_thread.daemon = True
    current_thread.start()
    
    return jsonify(response)
# start_yolo2x2()

if __name__ == "__main__":
    mqtt_thread = threading.Thread(target=start_mqtt)
    mqtt_thread.daemon = True
    mqtt_thread.start()
    
    app.run(host="0.0.0.0", port=4444, debug=True)