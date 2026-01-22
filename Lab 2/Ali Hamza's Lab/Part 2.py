from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Image path relative to this script
image_path = "Content/download.jpeg"

if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found!")
    print("Please update the image_path variable with the correct path to your image.")
else:
    image = Image.open(image_path)
    image_array = np.array(image)

    height, width, channels = image_array.shape
    print(f"Height: {height}, Width: {width}, Channels: {channels}")

    # Display individual RGB channels
    plt.figure(figsize=(15, 5))

    for i, channel in enumerate(["Red", "Green", "Blue"]):
        plt.subplot(1, 3, i + 1)
        # Extract individual channel
        channel_data = image_array[:, :, i]
        plt.imshow(channel_data, cmap="gray")
        plt.axis("off")
        plt.title(f"{channel} Channel")

    plt.tight_layout()
    plt.show()
