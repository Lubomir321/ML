import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    carsAdv_csv = pd.read_csv('cars_advanced.csv', index_col=0)

    print(carsAdv_csv[carsAdv_csv['drives_right']])

    print(carsAdv_csv.loc[carsAdv_csv['cars_per_cap'] > 500, 'country'])

    print(carsAdv_csv.loc[(carsAdv_csv['cars_per_cap'] >= 10) & (carsAdv_csv['cars_per_cap'] <= 80), 'country'])

    print(carsAdv_csv[carsAdv_csv['cars_per_cap'].between(10,80), 'country'])

if __name__ == '__main__':
    main()