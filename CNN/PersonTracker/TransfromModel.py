from ultralytics import YOLO

# Load the trained YOLOv8 model from the specified .pt weights file.
# This path should point to the best checkpoint obtained during training.
model = YOLO('runs/detect/tracker_aug/weights/best.pt')

# Export the model to ONNX format for deployment.
# - format="onnx": export as ONNX graph.
# - opset=12: ONNX opset version (ensure compatibility with your runtime).
# - dynamic=False: use fixed input size (no dynamic axes).
# - imgsz=160: input image size (160 x 160); should match the training configuration.
model.export(format="onnx", opset=12, dynamic=False, imgsz=160)
