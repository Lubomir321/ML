import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

def main():
    df = pd.read_csv("music_clean.csv", index_col=0)

    X = df.drop(columns=["energy"])
    y = df["energy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    ridge_model = Ridge(alpha=0.1, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)

    y_pred_linear = linear_model.predict(X_test_scaled)
    y_pred_ridge = ridge_model.predict(X_test_scaled)

    print(f"Linear Regression Test Set RMSE: {root_mean_squared_error(y_test, y_pred_linear)}")
    print(f"Ridge Test Set RMSE: {root_mean_squared_error(y_test, y_pred_ridge)}")
if __name__ == '__main__':
    main()