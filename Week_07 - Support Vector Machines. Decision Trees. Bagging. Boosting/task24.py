import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split


def preprocess_bike_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)

    # Convert datetime to numeric features
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    df["month"] = pd.to_datetime(df["datetime"]).dt.month
    df["day"] = pd.to_datetime(df["datetime"]).dt.day
    df.drop(columns=["datetime"], inplace=True)

    X = df.drop(columns=["count"])
    y = df["count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train.values, X_test.values, y_train.values, y_test.values

def main():
    filepath = "bike_sharing.csv"
    X_train, X_test, y_train, y_test = preprocess_bike_data(filepath)

    gbr = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=2)

    gbr.fit(X_train, y_train)

    y_pred = gbr.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Test set RMSE: {rmse:.2f}")
if __name__ == '__main__':
    main()