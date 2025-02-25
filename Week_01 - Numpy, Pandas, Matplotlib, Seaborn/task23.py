import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def main():
    cars_csv = pd.read_csv('cars.csv')
    print(cars_csv)
    print("After setting first column as index:")
    cars_csv1 = pd.read_csv('cars.csv', index_col=0)
    print(cars_csv1)


if __name__ == '__main__':
    main()
