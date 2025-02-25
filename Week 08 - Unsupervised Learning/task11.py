import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    features = data.iloc[:, :-1]
    varieties = data.iloc[:, 7]


    learning_rates = [10, 50, 200]

    for rate in learning_rates:
        tsne = TSNE(learning_rate=rate, random_state=42)
        tsne_features = tsne.fit_transform(features)

        x = tsne_features[:, 0]
        y = tsne_features[:, 1]

        unique_varieties = sorted(varieties.unique())
        for variety in unique_varieties:
            idx = varieties == variety
            plt.scatter(x[idx], y[idx])
        plt.title(f't-SNE on the grain dataset')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()