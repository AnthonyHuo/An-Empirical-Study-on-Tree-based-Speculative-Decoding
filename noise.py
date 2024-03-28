from skimage import data, img_as_float
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

# Load an example image and convert to grayscale
original_image = img_as_float(rgb2gray(data.chelsea()))

# Generate zero-mean Gaussian random noise and add to the original image
def add_noise(image, mean=0, std=0.1):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return noisy_image

# Generate N noisy images and compute the averaged image
def average_noisy_images(image, N):
    """Generate N noisy images and return their average."""
    noisy_images = np.array([add_noise(image) for _ in range(N)])
    averaged_image = noisy_images.mean(axis=0)
    return averaged_image

# Plot the averaged image for different values of N
Ns = [3, 10, 20, 100]

plt.figure(figsize=(15, 10))

for i, N in enumerate(Ns, 1):
    averaged_image = average_noisy_images(original_image, N)
    plt.subplot(2, 2, i)
    plt.imshow(averaged_image, cmap='gray')
    plt.title(f'N = {N}')
    plt.axis('off')

plt.tight_layout()
plt.show()
plt.savefig('cat.png') 