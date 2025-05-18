import cv2

# HLS stream URL (must be public or authenticated properly)
url = 'https://streaming.udoncity.go.th:1935/live/Axis_IP754.stream/chunklist_w553778697.m3u8'

# Open the video stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed to open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received, end of stream or error")
        break

    # Show the frame
    cv2.imshow('HLS Stream', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()