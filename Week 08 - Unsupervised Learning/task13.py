import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


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

    correlation, _ = pearsonr(kernel_length, kernel_width)

    plt.scatter(kernel_width, kernel_length)
    plt.title(f'Pearson correlation: {np.corrcoef(kernel_length,kernel_width)[0,1]:.2f}')
    plt.xlabel('Width of kernel')
    plt.ylabel('Length of kernel')
    plt.show()

    #Yes, the hypothesis is true. The scatter plot shows a clear positive correlation between kernel length and width


if __name__ == '__main__':
    main()