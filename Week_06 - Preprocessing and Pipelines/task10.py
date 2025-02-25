import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def main():
    df = pd.read_json("music_dirty_missing_vals.txt")


    df['is_rock'] = (df['genre'] == 'Rock').astype(int)

    X = df.drop(columns=['genre', 'is_rock'])
    y = df['is_rock']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y, random_state=42)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=42))
    ])

    param_grid = {
        'logreg__solver': ['newton-cg', 'saga', 'lbfgs'],
        'logreg__C': np.linspace(0.001, 1.0, 10)
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    test_accuracy = grid_search.score(X_test, y_test)
    print(f"Tuned Logistic Regression Parameters: {best_params}")
    print(f"Accuracy: {test_accuracy}")

if __name__ == '__main__':
    main()