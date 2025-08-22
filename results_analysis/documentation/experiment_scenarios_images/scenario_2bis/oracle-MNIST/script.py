import os
import struct
import numpy as np
from PIL import Image
from pathlib import Path

# Paths
input_path = Path("train-images-idx3-ubyte")
output_dir = Path("mnist_images")
output_dir.mkdir(exist_ok=True)

# Read binary file
with open(input_path, "rb") as f:
    magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
    data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)

# Save images as PNG
for i,d in enumerate(data):  # limit to first 100 for now
    img = Image.fromarray(d, mode="L")
    img.save(output_dir / f"mnist_{i:05d}.png")
