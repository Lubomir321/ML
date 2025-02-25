import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    carsAdv_csv = pd.read_csv('cars_advanced.csv', index_col=0)

    for lab, row in carsAdv_csv.iterrows():
        print(f"{lab}: {row['cars_per_cap']}")

    
if __name__ == '__main__':
    main()