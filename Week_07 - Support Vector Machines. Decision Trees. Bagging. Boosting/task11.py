import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def main():
    digits = load_digits()

    X = digits.data
    y = (digits.target == 2).astype(int)

    param_grid = {'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}

    svc = SVC(kernel='rbf', C=1, random_state=42)

    grid_search = GridSearchCV(svc, param_grid, cv=5)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best CV parameters: {best_params}")
    print(f"Best CV accuracy: {best_score}")
if __name__ == '__main__':
    main()