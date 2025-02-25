import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

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

    print(data)
    features = data.iloc[:, :-1]

    inertia = []
    cluster_range = range(1, 7)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)

    plt.plot(cluster_range, inertia, marker='o')
    plt.title('Inertia per number of clusters')
    plt.xlabel('number of clusters, k')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.show()

    # A good number of clusters can be determined by the "elbow method".(A point after which no significant changes are made)
    # For this dataset, the elbow typically occurs at 3 clusters.
if __name__ == '__main__':
    main()