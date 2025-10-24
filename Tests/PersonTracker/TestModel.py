from ultralytics import YOLO  # Librería de modelos YOLOv8 (detección, segmentación, etc.)
import cv2                    # OpenCV: procesamiento de imágenes y acceso a cámara
import time                   # Para delays y control de flujo

# Cargar el modelo entrenado (el mejor durante el entrenamiento)
model = YOLO('runs/detect/tracker_aug/weights/best.pt')

# Iniciar captura de video desde la cámara (0 = cámara por defecto)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # En Linux, CAP_V4L2 mejora compatibilidad
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # Evita acumulación de frames viejos

# Verificar que la cámara se haya abierto correctamente
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("✅ Cámara iniciada correctamente. Presiona 'q' para salir.")
time.sleep(1.0)  # Esperar un segundo para estabilizar la cámara

# Crear la ventana de visualización solo una vez
cv2.namedWindow("YOLOv8 - Detección de Semáforo", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al leer el frame de la cámara.")
        continue

    # Redimensionar el frame para coincidir con el tamaño usado en el entrenamiento
    resized_frame = cv2.resize(frame, (160, 120))

    try:
        # Realizar inferencia sobre el frame capturado
        results = model(resized_frame, imgsz=160, conf=0.5)

        # Dibujar las detecciones en la imagen original
        annotated_frame = results[0].plot()

        # Mostrar el frame con anotaciones en la ventana pre-creada
        cv2.imshow("YOLOv8 - Detección de Semáforo", annotated_frame)

    except Exception as e:
        print(f"⚠️ Error durante la inferencia o visualización: {e}")
        continue

    # Salir del bucle si el usuario presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos al finalizar
cap.release()
cv2.destroyAllWindows()
