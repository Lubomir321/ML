import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    column_names = [
        "area",
        "perimeter",
        "compactness",
        "length_of_kernel",
        "width_of_kernel",
        "asymmetry_coefficient",
        "length_of_kernel_groove",
        "varieties"
    ]

    data = pd.read_csv("seeds_dataset.txt", sep="\\s+", header=None, names=column_names)

    kernel_features = data.iloc[:, [3, 4]]

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(kernel_features)

    pc1 = pca_features[:, 0]
    pc2 = pca_features[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(pc1, pc2, alpha=0.7)
    plt.title(f'Pearson correlation: {np.corrcoef(pc1,pc2)[0,1]:.2f}')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()