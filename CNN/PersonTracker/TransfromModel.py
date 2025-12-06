from ultralytics import YOLO

model = YOLO('runs/detect/tracker_aug/weights/best.pt')

model.export(format="onnx", opset=12, dynamic=False, imgsz=160)
