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
    features = data.iloc[:, :-1]
    varieties = data.iloc[:, 7]

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)

    clusters = kmeans.labels_
    cluster_mapping = {1: 'Kama wheat', 2: 'Rosa wheat', 3: 'Canadian wheat'}

    mapped_varieties = varieties.map(cluster_mapping)

    crosstab = pd.crosstab(index=clusters, columns=mapped_varieties, rownames=['Cluster'], colnames=['Varieties'])

    print(crosstab)

if __name__ == '__main__':
    main()