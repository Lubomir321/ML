import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

def main():
    df = pd.read_csv("auto.csv")

    X = df.drop(columns=["mpg", "origin"]) 
    y = df["mpg"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    tree = DecisionTreeRegressor(max_depth=8, random_state=3, min_samples_leaf=0.13)
    tree.fit(X_train, y_train)

    y_pred_tree = tree.predict(X_test)

    rmse_tree = root_mean_squared_error(y_test, y_pred_tree)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    y_pred_linear = linear_model.predict(X_test)

    rmse_linear = root_mean_squared_error(y_test, y_pred_linear)

    print(f"Regression Tree test set RMSE: {rmse_tree:.2f}")
    print(f"Linear Regression test set RMSE: {rmse_linear:.2f}")
if __name__ == '__main__':
    main()