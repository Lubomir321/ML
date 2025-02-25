import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def main():
    digits = load_digits()

    X = digits.data
    y = (digits.target == 2).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
        'loss': ['hinge', 'log_loss']
    }

    sgd = SGDClassifier(random_state=0)

    grid_search = GridSearchCV(estimator=sgd, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    best_model = grid_search.best_estimator_

    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred) #decided to try this function instead score

    print(f"Best CV params: {best_params}")
    print(f"Best CV accuracy: {best_cv_score:.4f}")
    print(f"Test accuracy of best grid search hypers: {test_accuracy}")
if __name__ == '__main__':
    main()