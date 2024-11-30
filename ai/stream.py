import cv2
import subprocess

# Path to your MP4 file
input_file = "cctv.mp4"

# RTMP server URL
rtmp_url = "rtmp://localhost:1935/stream/yolo"

# Open the video file with OpenCV
cap = cv2.VideoCapture(input_file)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Streaming with resolution {width}x{height} at {fps} FPS.")

# FFmpeg command for RTMP streaming
command = [
    "ffmpeg",
    "-y",  # Overwrite output file if it exists
    "-stream_loop", "-1",
    "-f", "rawvideo",  # Input format
    "-pix_fmt", "bgr24",  # Pixel format for OpenCV frames
    "-s", f"{width}x{height}",  # Frame size
    "-r", str(15),  # Frame rate
    "-i", "-",  # Input from pipe
    "-c:v", "libx264",  # Video codec
    "-preset", "veryfast",  # Encoding speed
    '-b:v', '4M',
    '-maxrate', '4M',
    '-bufsize', '8M',
    '-vf', 'scale=1920:1084',
    "-f", "flv",  # Output format
    rtmp_url
]

# Open FFmpeg process
process = subprocess.Popen(command, stdin=subprocess.PIPE)

# Read frames and write to FFmpeg
try:
    while True:
        ret, frame = cap.read()
        # resized_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        
        if not ret:
            print("End of video.")
            break

        # Write frame to FFmpeg stdin
        process.stdin.write(frame.tobytes())
except BrokenPipeError:
    print("Stream stopped or FFmpeg process ended.")
finally:
    # Cleanup
    cap.release()
    process.stdin.close()
    process.wait()
    print("Streaming finished.")