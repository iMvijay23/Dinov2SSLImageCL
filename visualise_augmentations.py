
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from dataset_preparation import transform1simple, transform2simple, transform1hard, transform2hard

# Load the Tiny ImageNet dataset
tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')

# Get 5 random images from the dataset
random_indices = np.random.randint(0, len(tiny_imagenet), size=5)
images = [Image.fromarray(np.array(tiny_imagenet[int(i)]['image'])) for i in random_indices]

# Choose the desired transforms
transform1 = transform1simple
transform2 = transform2simple

# Apply augmentations and convert tensors back to PIL Images
augmented_images1 = [transforms.ToPILImage()(transform1(image)) for image in images]
augmented_images2 = [transforms.ToPILImage()(transform2(image)) for image in images]

# Visualize original and augmented images
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for i, image in enumerate(images):
    axes[0, i].imshow(image)
    axes[0, i].set_title("Original Image {}".format(i + 1))

for i, augmented_image in enumerate(augmented_images1):
    axes[1, i].imshow(augmented_image)
    axes[1, i].set_title("Augmented Image {} (Transform 1)".format(i + 1))

for i, augmented_image in enumerate(augmented_images2):
    axes[2, i].imshow(augmented_image)
    axes[2, i].set_title("Augmented Image {} (Transform 2)".format(i + 1))

plt.tight_layout()
plt.show()
