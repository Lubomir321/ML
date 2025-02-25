import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    X = df[['mean radius', 'mean concave points']]  # Select features by name
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    tree = DecisionTreeClassifier(max_depth=6, random_state=1)
    tree.fit(X_train, y_train)

    y_test_pred = tree.predict(X_test)

    first_5_predictions = y_test_pred[:5]
    print(f"First 5 predictions: {list(first_5_predictions)}")

    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test set accuracy: {round(test_accuracy, 2)}")
if __name__ == '__main__':
    main()