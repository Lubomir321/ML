import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA

def main():
    data = pd.read_csv("lcd-digits.csv", header=None).values

    first_image = data[0].reshape(13, 8)
    plt.title("First Image")
    plt.imshow(first_image, cmap="gray")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    nmf = NMF(n_components=7, random_state=42)
    nmf_components = nmf.fit_transform(data)
    nmf_basis = nmf.components_

    for i, component in enumerate(nmf_basis):
        plt.subplot(1, 7, i + 1) #instead of making fig ax .....
        plt.imshow(component.reshape(13, 8), cmap="gray")
        plt.axis("off")
    plt.suptitle("Features learned by NMF")
    plt.tight_layout()
    plt.show()

    pca = PCA(n_components=7, random_state=42)
    pca_components = pca.fit_transform(data)
    pca_basis = pca.components_

    for i, component in enumerate(pca_basis):
        plt.subplot(1, 7, i + 1)
        plt.imshow(component.reshape(13, 8), cmap="gray")
        plt.axis("off")
    plt.suptitle("Features learned by PCA")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()