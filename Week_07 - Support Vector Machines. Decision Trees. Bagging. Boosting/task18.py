import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder

def main():
    # Load the dataset
    df = pd.read_csv("auto.csv")

    df = pd.get_dummies(df, columns=["origin"], drop_first=True)

    X = df.drop(columns=["mpg"])
    y = df["mpg"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    tree = DecisionTreeRegressor(max_depth=4, random_state=1, min_samples_leaf=0.26)
    tree.fit(X_train, y_train)

    y_train_pred = tree.predict(X_train)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)

    cv_scores = cross_val_score(tree, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores).mean()

    print("Dataset X:\n")
    print(X)

    print(f"\n10-fold CV RMSE: {cv_rmse:.2f}")
    print(f"Train RMSE: {train_rmse:.2f}")

    # The decision tree suffers from a high bias problem. 
    # Becasue of the constraints on the tree depth
if __name__ == '__main__':
    main()