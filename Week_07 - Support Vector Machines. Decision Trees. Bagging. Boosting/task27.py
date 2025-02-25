import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import root_mean_squared_error
from typing import Tuple

def main():
    def preprocess_bike_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = pd.read_csv(filepath)

        df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
        df["month"] = pd.to_datetime(df["datetime"]).dt.month
        df["day"] = pd.to_datetime(df["datetime"]).dt.day
        df.drop(columns=["datetime"], inplace=True)

        X = df.drop(columns=["count"])
        y = df["count"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        return X_train.values, X_test.values, y_train.values, y_test.values


    filepath = "bike_sharing.csv"
    X_train, X_test, y_train, y_test = preprocess_bike_data(filepath)

    rf = RandomForestRegressor(random_state=2)
    param_grid = {
        "n_estimators": [100, 350, 500],
        "max_features": ["log2", "sqrt", None],
        "min_samples_leaf": [2, 10, 30],
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Test set RMSE: {test_rmse:.3f}")

if __name__ == '__main__':
    main()
