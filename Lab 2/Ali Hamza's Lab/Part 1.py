from PIL import Image
import numpy as np
import os

# Image path relative to this script
image_path = 'Content/download.jpeg'

if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found!")
    print("Please update the image_path variable with the correct path to your image.")
else:
    image = Image.open(image_path)
    image_array = np.array(image)
    
    print(f"Image Shape: {image_array.shape}")
    print("Image opened successfully")