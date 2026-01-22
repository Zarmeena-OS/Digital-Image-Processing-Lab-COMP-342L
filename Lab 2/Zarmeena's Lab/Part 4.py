from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Use relative path for portability
image_path = os.path.join(os.path.dirname(__file__), 'Content', 'download.jpg')

# Open and convert to array efficiently, then close image to free memory
with Image.open(image_path) as image:
    image_array = np.array(image)

# Extract image dimensions
height, width, channels = image_array.shape
print(f"Height: {height}, Width: {width}, Channels: {channels}")

# Display RGB channels separately
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
channel_names = ['Red', 'Green', 'Blue']

for i, (channel_name, ax) in enumerate(zip(channel_names, axes)):
    ax.imshow(image_array[:, :, i], cmap='gray')
    ax.set_title(f'{channel_name} Channel', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()