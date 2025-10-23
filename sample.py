import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np
from PIL import Image

digits = load_digits()
index = 3  # change between 0–1796 to get different digits
data = digits.images[index]

plt.imshow(data, cmap='gray')
plt.title(f"Digit Label: {digits.target[index]}")
plt.axis('off')
plt.show()

# Save it as 8x8 grayscale for your Streamlit app
scaled = (data / data.max()) * 255
img = Image.fromarray(scaled.astype('uint8'))
img = img.convert('L')
img.save('digit_sample.png')

print("Saved: digit_sample.png (Digit:", digits.target[index], ")")