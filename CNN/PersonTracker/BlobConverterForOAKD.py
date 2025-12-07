#!/usr/bin/env python3
import blobconverter
from pathlib import Path

onnx_path = Path("runs/detect/tracker_aug/weights/best.onnx")

blob_path = blobconverter.from_onnx(
    model=str(onnx_path),
    data_type="FP16", 
    shaves=6,
    use_cache=False,
    output_dir="models"
)

print("Blob generated in:", blob_path)
