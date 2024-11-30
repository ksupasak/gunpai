import cv2
import numpy as np

# Function to decode H264 NAL units to raw frames (unchanged)

def decode_h264(data):
    """Decodes H264 NAL units to raw frames.

    Args:
        data: Bytes object containing H264 NAL units.

    Returns:
        A list of decoded frames as numpy arrays.
    """
    frames = []
    nal_start_code = b'\x00\x00\x00\x01'
    nal_starts = [0] + [i for i in range(len(data) - len(nal_start_code) + 1) 
                       if data[i:i + len(nal_start_code)] == nal_start_code]
    for i in range(len(nal_starts) - 1):
        nal_unit = data[nal_starts[i]:nal_starts[i + 1]]
        # Decode NAL unit using OpenCV
        frame = cv2.imdecode(np.frombuffer(nal_unit, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    return frames

# Initialize CUDA (if available)
try:
    cv2.cuda.setDevice(0)  # Use the first available GPU
    cuda_enabled = True
    # Create CUDA-enabled objects
    cuda_previous_frame = None
    cuda_frame_delta = cv2.cuda_GpuMat()
    cuda_thresh = cv2.cuda_GpuMat()
except:
    cuda_enabled = False
    print("CUDA not available. Running on CPU.")

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for default camera
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Initialize previous frame
previous_frame = None
motion_detected = False

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    if cuda_enabled:
        # Upload frame to GPU
        cuda_frame = cv2.cuda_GpuMat(frame)

        # Convert frame to grayscale on GPU
        cuda_gray = cv2.cuda.cvtColor(cuda_frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur on GPU
        cuda_gray = cv2.cuda.GaussianBlur(cuda_gray, (21, 21), 0)

        if cuda_previous_frame is None:
            cuda_previous_frame = cuda_gray
            continue

        # Calculate frame difference on GPU
        cv2.cuda.absdiff(cuda_previous_frame, cuda_gray, cuda_frame_delta)
        # Apply thresholding on GPU
        cv2.cuda.threshold(cuda_frame_delta, 25, 255, cv2.THRESH_BINARY, cuda_thresh)
        # Dilate on GPU
        cv2.cuda.dilate(cuda_thresh, None, iterations=2, dst=cuda_thresh)

        # Download thresholded image from GPU
        thresh = cuda_thresh.download()

        # Update previous frame
        cuda_previous_frame = cuda_gray

    else:  # CPU processing
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if previous_frame is None:
            previous_frame = gray
            continue

        # Calculate frame difference
        frame_delta = cv2.absdiff(previous_frame, gray)
        # Apply thresholding to detect motion
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        # Dilate the thresholded image to fill holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Update previous frame
        previous_frame = gray

    # Find contours of motion
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check for motion
    for c in cnts:
        if cv2.contourArea(c) < 500:  # Adjust minimum area as needed
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    # Display the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Print motion detection status
    if motion_detected:
        print("Motion detected!")
        motion_detected = False  # Reset for next frame

# Cleanup
cap.release()
cv2.destroyAllWindows()