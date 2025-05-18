import cv2

# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(1)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to continuously capture and display frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is not captured successfully, exit the loop
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Display the captured frame
    cv2.imshow('Webcam Video', frame)
    print(f"{frame}")
    
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()