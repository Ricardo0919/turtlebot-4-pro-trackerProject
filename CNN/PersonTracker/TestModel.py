# ===============================================================================
# Project: Turtlebot 4 Pro - SScript for Testing a YOLOv8 Model for Person Tracking
# Student Project
# Date: December 7th, 2025
# Students:
#   - Sergi Fernandez Mendez
#   - Ricardo Sierra Roa
# ===============================================================================

from ultralytics import YOLO  # YOLOv8 models for detection, segmentation, etc.
import cv2                    # OpenCV: image processing and camera access
import time                   # For delays and simple timing

# Load the trained YOLOv8 model (best checkpoint from training)
model = YOLO('runs/detect/tracker_aug/weights/best.pt')

# Open video capture from the default camera (index 0)
# On Linux, CAP_V4L2 often improves compatibility with USB cameras.
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# Reduce internal buffer to avoid latency caused by queued old frames
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Verify that the camera was opened successfully
if not cap.isOpened():
    print("Error: could not open camera.")
    exit(1)

print("Camera started successfully. Press 'q' to exit.")
# Small delay to let the camera auto-exposure and focus stabilize
time.sleep(1.0)

# Create the display window only once
cv2.namedWindow("Tracker person", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to read frame from camera.")
        continue

    # Resize the frame to match the training resolution (160 x 120)
    resized_frame = cv2.resize(frame, (160, 120))

    try:
        # Run inference on the captured frame
        # imgsz=160 sets the internal inference size, consistent with training
        results = model(resized_frame, imgsz=160, conf=0.5)

        # Draw detections on the frame (bounding boxes, labels, confidences)
        annotated_frame = results[0].plot()

        # Show the annotated frame in the pre-created window
        cv2.imshow("Tracker person", annotated_frame)

    except Exception as e:
        print(f"Warning: error during inference or visualization: {e}")
        continue

    # Exit loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources when finished
cap.release()
cv2.destroyAllWindows()
