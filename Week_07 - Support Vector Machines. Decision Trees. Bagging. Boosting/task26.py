import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score


def preprocess(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = [
        "age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp",
        "alb", "a/g_ratio", "has_liver_disease"
    ]
    df = pd.read_csv(filepath, header=None, names=columns)

    df.fillna(df.mode().iloc[0], inplace=True)

    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

    df["has_liver_disease"] = df["has_liver_disease"].map({2: 0, 1: 1})

    X = df.drop(columns=["has_liver_disease"])
    y = df["has_liver_disease"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    return X_train.values, X_test.values, y_train.values, y_test.values


filepath = "indian_liver_patient_dataset.csv"
X_train, X_test, y_train, y_test = preprocess(filepath)

dt = DecisionTreeClassifier(random_state=1)
param_grid = {
    "max_depth": [2, 3, 4],
    "min_samples_leaf": [0.12, 0.14, 0.16, 0.18],
}

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring="roc_auc")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Test set ROC AUC: {test_roc_auc:.3f}")
