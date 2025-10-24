from ultralytics import YOLO  # Librería principal para usar modelos YOLOv8
import torch                  # PyTorch: backend que usa YOLOv8 para entrenamiento
import multiprocessing        # Necesario en Windows, no afecta en Linux

def main():
    # Verifica si CUDA (GPU) está disponible para acelerar el entrenamiento
    print("CUDA disponible:", torch.cuda.is_available())

    # Carga del modelo YOLOv8 preentrenado ("s" = small)
    # Este modelo servirá como base y se ajustará al dataset específico de semáforos
    model = YOLO('yolov8s.pt')

    # Reentrenamiento del modelo con data augmentation personalizado
    model.train(
        data='data.yaml',                    # Configuración del dataset
        epochs=50,                           # Número de épocas
        imgsz=160,                           # Tamaño de las imágenes
        batch=32,                            # Tamaño del lote
        device=0,                            # GPU
        name='tracker_aug',  # Nombre
        augment=True,                        # Activar data augmentation
        degrees=10,                          # Rotación aleatoria ±10°
        scale=0.5,                           # Escalado aleatorio (0.5 a 1.5)
        shear=10,                            # Shear horizontal/vertical ±10°
        perspective=0.001,                   # Distorsión de perspectiva leve
        flipud=0.0,                          # No voltear verticalmente
        fliplr=0.5,                          # 50% de voltear horizontal
        hsv_h=0.015,                         # Variación de tono
        hsv_s=0.7,                           # Variación de saturación
        hsv_v=0.4                            # Variación de brillo
    )

# Punto de entrada del script
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
