import cv2
import numpy as np

# Initialize video capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Parameters for Farneback optical flow
farneback_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                        poly_n=5, poly_sigma=1.2, flags=0)

# Read the first frame and convert it to grayscale
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    # Capture the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback's method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **farneback_params)

    # Visualize flow with color
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    # Use angle to set the hue and magnitude to set the value (brightness)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR color space
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Overlay optical flow on the original frame
    overlay = cv2.addWeighted(frame, 0.7, flow_bgr, 0.3, 0)

    # Display the combined image
    cv2.imshow('Optical Flow Overlay', overlay)

    # Update the previous frame
    prev_gray = gray

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()