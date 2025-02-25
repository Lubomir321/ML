import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import scipy.stats as st

def main():
    df = pd.read_csv("advertising_and_sales_clean.csv")

    X = df.drop(columns=['sales', 'influencer'])
    y = df['sales']


    kf = KFold(n_splits=6, shuffle=True, random_state=5)

    linear_reg = LinearRegression()

    cv_results = cross_val_score(linear_reg, X, y, cv=kf)

    mean_score = np.mean(cv_results)
    std_score = np.std(cv_results)
    confidence_interval = np.quantile(cv_results, [0.025, 0.975])

    print(f"Mean: {mean_score}")
    print(f"Standard Deviation: {std_score}")
    print(f"95% Confidence Interval: {confidence_interval}")

    plt.plot(range(1, 7), cv_results)
    plt.title("R^2 per 6-fold split")
    plt.xlabel("# Split")
    plt.ylabel("R^2")
    plt.ylim([0.998875, 0.999100])
    plt.yticks(np.linspace(0.998900, 0.999100, 9))
    plt.show()

if __name__ == '__main__':
    main()