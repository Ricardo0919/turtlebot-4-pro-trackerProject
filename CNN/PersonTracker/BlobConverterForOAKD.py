#!/usr/bin/env python3
import blobconverter
from pathlib import Path

# Ajusta esta ruta a tu ONNX exportado
onnx_path = Path("runs/detect/tracker_aug/weights/best.onnx")

blob_path = blobconverter.from_onnx(
    model=str(onnx_path),
    data_type="FP16",   # FP16 para MyriadX
    shaves=6,           # núcleos de NPU (6–7 suele estar bien)
    use_cache=False,
    output_dir="models" # carpeta donde quieres el .blob
)

print("✅ Blob generado en:", blob_path)
