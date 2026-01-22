from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Use relative path for portability
image_path = os.path.join(os.path.dirname(__file__), 'Content', 'download.jpg')

# Open and convert to array efficiently, then close image to free memory
with Image.open(image_path) as image:
    image_array = np.array(image)

print(f"Image shape: {image_array.shape}")
print("Image opened successfully")

# Display the image
plt.imshow(image_array)
plt.axis('off')  # Hide axes for cleaner display
plt.title('Image Display')
plt.show()
