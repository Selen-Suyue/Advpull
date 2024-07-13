from PIL import Image
import numpy as np


image_path = "output_image.jpg"
image = Image.open(image_path)

image_array = np.array(image)

scaled_image_array = image_array * 255

scaled_image = Image.fromarray(scaled_image_array.astype(np.uint8))

scaled_image.save("scaled_image.jpg")


