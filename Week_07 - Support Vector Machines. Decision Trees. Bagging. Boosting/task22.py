import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

def preprocess(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(filepath, parse_dates=["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    df = df.drop(columns=["datetime"])

    X = df.drop(columns=["count"])
    y = df["count"]

    return X, y

def main():
    filepath = "bike_sharing.csv"
    X, y = preprocess(filepath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    rf = RandomForestRegressor(n_estimators=25, random_state=2)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Test set RMSE: {rmse:.2f}")

    feature_importances = rf.feature_importances_
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importances)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importances from Random Forest Model")
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()