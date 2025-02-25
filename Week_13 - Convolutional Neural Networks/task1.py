import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random


def plot_random_images(dataset, num_images=6):
    random_indices = random.sample(range(len(dataset)), num_images)
    class_names = dataset.classes

    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(random_indices):
        image, label = dataset[idx]
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"{class_names[label]}")
    plt.title("The Clouds dataset")
    plt.tight_layout()
    plt.show()


def main():
    dataset_train = ImageFolder('clouds/clouds_train')
    
    plot_random_images(dataset_train)

if __name__ == "__main__":
    main()