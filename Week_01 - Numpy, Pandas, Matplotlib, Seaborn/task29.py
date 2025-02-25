import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    carsAdv_csv = pd.read_csv('cars_advanced.csv', index_col=0)

    print(carsAdv_csv)

    carsAdv_csv['COUNTRY'] = carsAdv_csv['country'].str.upper()

    print(carsAdv_csv)


if __name__ == '__main__':
    main()