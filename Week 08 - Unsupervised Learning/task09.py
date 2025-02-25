import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('eurovision_voting.csv', index_col=0)

    voting_data = data.values

    mergings = linkage(voting_data, method='single')

    plt.figure(figsize=(15, 10))
    dendrogram(mergings, labels=data.index, leaf_rotation=90, leaf_font_size=6)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()