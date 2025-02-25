import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

    return X_train.values, X_test.values, y_train.values, y_test.values

def main():
    filepath = "indian_liver_patient_dataset.csv"
    X_train, X_test, y_train, y_test = preprocess(filepath)

    weak_learner = DecisionTreeClassifier(max_depth=2, random_state=1)
    ada_boost = AdaBoostClassifier(estimator=weak_learner, n_estimators=180, random_state=1)

    ada_boost.fit(X_train, y_train)

    y_pred_prob = ada_boost.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Test set ROC AUC: {roc_auc:.2f}")
if __name__ == '__main__':
    main()