import cv2
import torch
import coremltools as ct

import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLOv8 model
# model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # Choose your desired YOLOv8 variant

#model = YOLO("yolov8n-seg.pt")
#model = YOLO("yolov8m.pt")
# model = YOLO("yolov8n-seg.onnx")
# Load the YOLOv8 model (you'll need to convert it to Core ML first)

# Load the YOLOv8x model
# model = YOLO('yolov8x.pt')
# pytorch_model = model.model
#
#
#
# # Export the model to CoreML format
# model.export(format="coreml")  # creates 'yolo11n.mlpackage'
#
# # Load the exported CoreML model
# coreml_model = YOLO("yolov8x.mlpackage")



# # Create a sample input
# sample_input = torch.randn(1, 3, 640, 640)  # Example input shape
#
# # Trace the model to create a TorchScript object
# traced_model = torch.jit.trace(pytorch_model, sample_input)
# # Convert the model to Core ML
# # Convert the traced TorchScript model to Core ML
# mlmodel = ct.convert(
#     traced_model,
#     source="pytorch",
#     inputs=[ct.ImageType(name="image", shape=(1, 3, 640, 640), scale=1/255.0)],
# )
#
# # Save the Core ML model
# mlmodel.save('yolov8x.mlmodel')

try:
    model = ct.models.MLModel('yolov8x.mlpackage')  # Replace with your Core ML model file
except Exception as e:
    print(f"Error loading Core ML model: {e}")
    exit()

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# Enable GPU acceleration if available
if torch.cuda.is_available():
    model.cuda() 
    
device = "cpu"

# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify video file path

while(True):
    # Read a frame from the video
    ret, frame = cap.read()

 # Preprocess the frame (resize and normalize)
    resized_frame = cv2.resize(frame, (640, 640))  # Resize to match model input shape
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]

    # Convert to Core ML compatible format (CVPixelBuffer)
    # image_input = ct.ImageType(name="image", shape=(1, 3, 640, 640))
    
    # Convert the NumPy array to a CVPixelBuffer
    # coreml_input = image_input.from_numpy_array(normalized_frame)
    pil_image = Image.fromarray(np.uint8(normalized_frame * 255))  # Convert to PIL Image
    
    # coreml_input = ct.pixel_buffer_from_numpy_array(normalized_frame)
    
    try:
        output = model.predict({'image': pil_image}) 
    except Exception as e:
        print(f"Error during prediction: {e}")
        continue
    print(output)
 # # Assuming the output is a dictionary with 'boxes', 'labels', and 'scores' keys
    boxes = output['coordinates']  # Array of bounding boxes (x1, y1, x2, y2)
    # labels = output['labels']  # Array of class labels (integers)
    # scores = output['scores']  # Array of confidence scores
    #
    # # Plot the bounding boxes and labels on the original frame
    for box in zip(boxes):
        x1, y1, x2, y2 = map(int, box[0] * 640)  # Convert box coordinates to integers
        
        cv2.rectangle(resized_frame, (x1-(int)(x2/2), y1-(int)(y2/2)), ((int)(x2/2)+x1, (int)(y2/2)+y1), (0, 255, 0), 2)  # Draw green rectangle
    #     cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # annotated_frame = output[0].plot()
    # Display the annotated frame
    cv2.imshow('YOLOv8 Object Detection', resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()