import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    carsAdv_csv = pd.read_csv('cars_advanced.csv', index_col=0)
    print(carsAdv_csv)
    uppercase_countries = []

    for lab, row in carsAdv_csv.iterrows():
        upper_country = row['country'].upper()
        uppercase_countries.append(upper_country)

    carsAdv_csv['COUNTRY'] = uppercase_countries

    print(carsAdv_csv)

if __name__ == '__main__':
    main()
