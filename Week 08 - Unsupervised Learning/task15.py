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

    kernel_length = data.iloc[:, 3]
    kernel_width = data.iloc[:, 4]
    kernel_features = np.column_stack((kernel_length, kernel_width))

    pca = PCA(n_components=2)
    pca.fit(kernel_features)



    plt.scatter(kernel_width, kernel_length)
    plt.title('First Principal Component')
    plt.xlabel('Kernel Width')
    plt.ylabel('Kernel Length')
    plt.arrow(3.06, 5.3, 0.8, 1.1, color='red', width=0.01, head_width=0.1, head_length=0.1) #hardcoded for specific data
    #maybe compute random point from the lowest part and random point of the upper part and find the distance btween them
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()