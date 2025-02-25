import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, KFold

def main():
    df = pd.read_csv('diabetes_clean.csv')

    X = df.drop(columns='diabetes')
    y = df['diabetes']

    log_reg = LogisticRegression(random_state=42)

    param_grid = {
        'penalty': ['l1', 'l2'],
        'tol': np.linspace(0.0001, 1.0, 50),
        'C': np.linspace(0.1, 1.0, 50),
        'class_weight': ['balanced', {0: 0.8, 1: 0.2}]
    }

    kf = KFold(n_splits=6, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(log_reg, param_distributions=param_grid, n_iter=3, cv=kf, random_state=42)
    
    random_search.fit(X, y)

    print("Tuned Logistic Regression Parameters:", random_search.best_params_)
    print("Tuned Logistic Regression Best Accuracy Score:", random_search.best_score_)
if __name__ == '__main__':
    main()