# Dockerfile for YOLOv8 Server
FROM python:3.8-slim

WORKDIR /app

# Install dependencies
RUN pip install ultralytics fastapi uvicorn

# Copy YOLOv8 server code
COPY yolov8_server.py .

# Run FastAPI server
CMD ["uvicorn", "yolov8_server:app", "--host", "0.0.0.0", "--port", "8000"]