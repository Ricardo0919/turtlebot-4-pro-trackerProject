# ===============================================================================
# Project: Turtlebot 4 Pro - Script for Training a YOLOv8 Model with Data Augmentation
# Student Project
# Date: December 7th, 2025
# Students:
#   - Sergi Fernandez Mendez
#   - Ricardo Sierra Roa
# ===============================================================================

from ultralytics import YOLO  # Main library for working with YOLOv8 models
import torch                  # PyTorch backend used by YOLOv8 for training
import multiprocessing        # Needed on Windows; harmless on Linux


def main():
    # Check if CUDA (GPU) is available to accelerate training
    print("CUDA available:", torch.cuda.is_available())

    # Load a pretrained YOLOv8 small model ("s" = small, lightweight backbone)
    # This model will be fine-tuned on the custom dataset.
    model = YOLO('yolov8s.pt')

    # Fine-tune the model with custom data augmentation
    model.train(
        data='data.yaml',        # Dataset configuration file
        epochs=50,               # Number of training epochs
        imgsz=160,               # Input image size (160 x 160)
        batch=32,                # Batch size
        device=0,                # GPU index (0 = first GPU)
        name='tracker_aug',      # Run name (used for logs and weights folder)
        augment=True,            # Enable built-in data augmentation

        # Custom augmentation parameters
        degrees=10,              # Random rotation range ±10°
        scale=0.5,               # Random scaling factor (0.5 to 1.5 internally)
        shear=10,                # Random shear (horizontal/vertical) ±10°
        perspective=0.001,       # Slight perspective distortion
        flipud=0.0,              # Probability of vertical flip (0 = disabled)
        fliplr=0.5,              # Probability of horizontal flip (50%)
        hsv_h=0.015,             # Hue variation
        hsv_s=0.7,               # Saturation variation
        hsv_v=0.4                # Value (brightness) variation
    )


# Script entry point
if __name__ == '__main__':
    # Required on Windows to safely spawn training processes
    multiprocessing.freeze_support()
    main()
