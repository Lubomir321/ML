import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def main():
  digits = load_digits()

  X = digits.data
  y = (digits.target == 2).astype(int)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

  param_grid = {
      'C': [0.1, 1, 10],
      'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]
  }

  svc = SVC(kernel='rbf', random_state=42)

  grid_search = GridSearchCV(svc, param_grid=param_grid, cv=5)
  grid_search.fit(X_train, y_train)

  best_params = grid_search.best_params_
  best_cv_score = grid_search.best_score_

  best_model = grid_search.best_estimator_

  y_test_pred = best_model.predict(X_test)

  print(f"Best parameters: {best_params}")
  print(f"Best cross-validation accuracy: {best_cv_score}")
  print(f"Test set accuracy: {grid_search.score(X_test,y_test)}")

  #smaller datasets tend to require simpler models (lower gamma values).
  
if __name__ == '__main__':
    main()