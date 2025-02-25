import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    df = pd.read_csv('advertising_and_sales_clean.csv')

    print(df.loc[[0, 1]])

    X_sales = df[['sales']]
    X_radio = df[['radio']]
    X_tv = df[['tv']]
    X_social_media = df[['social_media']]

    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].scatter(X_tv, X_sales)
    ax[0].set_xlabel("tv")
    ax[0].set_ylabel("Sales")
    ax[0].set_title("tv vs sales")

    ax[1].scatter(X_radio, X_sales)
    ax[1].set_xlabel("radio")
    ax[1].set_ylabel("Sales")
    ax[1].set_title("radio vs sales")

    ax[2].scatter(X_social_media, X_sales)
    ax[2].set_xlabel("social_media")
    ax[2].set_ylabel("Sales")
    ax[2].set_title("social_media vs sales")
    plt.tight_layout()
    plt.show()

    #tv vs sales
    reg = LinearRegression()
    reg.fit(X_radio, X_sales)

    predictions = reg.predict(X_radio)
    print("Feature with highest correlation (from visual inspection): tv")
    flattened = np.array(predictions[0:5])
    print(f"First five predictions: {flattened.flatten()}")

    plt.scatter(X_radio, X_sales)
    plt.plot(X_radio, predictions, color='red', linewidth=2, zorder=2)
    plt.ylabel('Sales ($)')
    plt.xlabel('Radio Expenditure ($)')
    plt.title('Rlationship between radio expenditures and sales')
    plt.show()
if __name__ == '__main__':
    main()