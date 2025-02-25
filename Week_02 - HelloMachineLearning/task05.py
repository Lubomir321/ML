import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def main():
    class KNN:
        def __init__(self, n_neighbors=5):
            self.n_neighbors = n_neighbors

        def fit(self, X_train, y_train):
            self.X_train = X_train
            self.y_train = y_train

        def predict(self, X_test):
            predictions = []
            for x in X_test:
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
                
                neighbors_indices = np.argsort(distances)[:self.n_neighbors]
                
                nearest_labels = self.y_train[neighbors_indices]
                
                most_common = Counter(nearest_labels).most_common(1)[0][0]

                predictions.append(most_common)
            return np.array(predictions)

        def score(self, X_test, y_test):
            predictions = self.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            return accuracy


    df_churn = pd.read_csv('telecom_churn_clean.csv')

    X = df_churn.drop(columns=['Unnamed: 0', 'churn']).values

    y = df_churn['churn'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    custom_knn = KNN(n_neighbors=5)
    custom_knn.fit(X_train, y_train)


    custom_knn_accuracy = custom_knn.score(X_test, y_test)


    sklearn_knn = KNeighborsClassifier(n_neighbors=5)
    sklearn_knn.fit(X_train, y_train)


    sklearn_knn_accuracy = sklearn_knn.score(X_test, y_test)


    print(f"Custom KNN Accuracy: {custom_knn_accuracy:.4f}")
    print(f"Sklearn KNN Accuracy: {sklearn_knn_accuracy:.4f}")

if __name__ == '__main__':
    main()