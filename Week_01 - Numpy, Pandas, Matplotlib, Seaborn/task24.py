import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():

    cars_csv = pd.read_csv('cars.csv', index_col=0)

    print(cars_csv['country'])

    print(cars_csv[['country']])

    print(cars_csv[['country', 'drives_right']])

    print(cars_csv[0:3])

    print(cars_csv[3:6])

    carsadv_csv = pd.read_csv('cars_advanced.csv', index_col = 0)

    print(carsadv_csv.loc['JPN'])

    print(carsadv_csv.loc[['AUS', 'EG']])

    print(carsadv_csv.loc[['MOR'], ['drives_right']])

    print(carsadv_csv.loc[['MOR', 'RU'], ['country','drives_right']])


if __name__ == '__main__':
    main()