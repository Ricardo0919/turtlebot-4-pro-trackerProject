# ===============================================================================
# Project: Turtlebot 4 Pro - Script for Testing a YOLOv8 ONNX Model for Person Tracking
# Student Project
# Date: December 7th, 2025
# Students:
#   - Sergi Fernandez Mendez
#   - Ricardo Sierra Roa
# ===============================================================================

from ultralytics import YOLO  # Ultralytics also supports running ONNX models
import cv2
import time

# Load the exported ONNX model (must match the training configuration)
model = YOLO('runs/detect/tracker_aug/weights/best.onnx')

# Open video capture from default camera (index 0)
# On Linux, CAP_V4L2 often improves compatibility with USB cameras.
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# Reduce internal buffer to avoid latency from queued old frames
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: could not open camera.")
    exit(1)

print("Camera started successfully. Press 'q' to exit.")
time.sleep(1.0)

cv2.namedWindow("Tracker person (ONNX)", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: failed to read frame from camera.")
        continue

    # Resize the frame to the expected input size
    # (160 x 128 here; must be consistent with training/export settings)
    resized_frame = cv2.resize(frame, (160, 128))

    try:
        # Run ONNX inference through Ultralytics
        # imgsz=160 sets the internal square inference size,
        # while the input is already resized above.
        results = model(resized_frame, imgsz=160, conf=0.5)

        # Get annotated frame (bounding boxes, labels, scores)
        annotated_frame = results[0].plot()

        cv2.imshow("Tracker person (ONNX)", annotated_frame)

    except Exception as e:
        print(f"Warning: error during inference or visualization: {e}")
        continue

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
