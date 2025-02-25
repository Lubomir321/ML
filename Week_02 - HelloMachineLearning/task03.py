import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def main():
    df_churn = pd.read_csv('telecom_churn_clean.csv')
    X = df_churn[['account_length', 'customer_service_calls']].values
    y = df_churn['churn'].values

    print(X.shape, y.shape)

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X, y)

    X_new = np.array([[30.0, 17.5],
                    [107.0, 24.1],
                    [213.0, 10.9]])
    print(X_new.shape)

    predictions = knn.predict(X_new)
    print(f'{predictions=}')

if __name__ == '__main__':
    main()