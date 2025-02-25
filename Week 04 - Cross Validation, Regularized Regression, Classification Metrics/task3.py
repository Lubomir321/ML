import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def main():
    df = pd.read_csv("advertising_and_sales_clean.csv")

    X = df.drop(columns=['sales', 'influencer'])
    y = df['sales']

    lasso = Lasso(alpha=1.0) 
    lasso.fit(X, y)

    coefficients = {feature: float(np.round(coef, 4)) for feature, coef in zip(X.columns, lasso.coef_)}
    print("Lasso coefficients per feature:", coefficients)

    plt.bar(coefficients.keys(), coefficients.values())
    plt.title("Feature importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()
    #Which is the most important to predict sales? -> tv

if __name__ == '__main__':
    main()