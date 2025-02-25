import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

def main():
    df = pd.read_csv('advertising_and_sales_clean.csv')

    X = df.drop(['sales', 'influencer'], axis=1)
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    prediction = reg.predict(X_test)

    print(f"Predictions: {prediction[:2]}")
    print(f"Actual Values: {y_test.values[:2]}")

    print(f"R^2: {reg.score(X_test, y_test)}")
    
    rmse = root_mean_squared_error(y_test, prediction)
    print(f"RMSE: {rmse}")

if __name__ == '__main__':
    main()
