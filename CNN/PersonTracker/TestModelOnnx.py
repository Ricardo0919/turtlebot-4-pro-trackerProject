#!/usr/bin/env python3
from ultralytics import YOLO  # Ultralytics también soporta modelos ONNX
import cv2
import time

model = YOLO('runs/detect/tracker_aug/weights/best.onnx')

# Iniciar captura de video desde la cámara (0 = cámara por defecto)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # En Linux, CAP_V4L2 mejora compatibilidad
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # Evita acumulación de frames viejos

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("✅ Cámara iniciada correctamente. Presiona 'q' para salir.")
time.sleep(1.0)

cv2.namedWindow("Tracker person (ONNX)", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al leer el frame de la cámara.")
        continue

    # Redimensionar al tamaño esperado (Ultralytics ya ajustó a [128, 160] internamente)
    resized_frame = cv2.resize(frame, (160, 128))

    try:
        # Inferencia con el modelo ONNX
        results = model(resized_frame, imgsz=160, conf=0.5)

        # Dibujar anotaciones
        annotated_frame = results[0].plot()

        cv2.imshow("Tracker person (ONNX)", annotated_frame)

    except Exception as e:
        print(f"⚠️ Error durante la inferencia o visualización: {e}")
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
