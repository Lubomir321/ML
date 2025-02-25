import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from typing import Tuple

def preprocess(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
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

    return X, y

def main():
    filepath = "indian_liver_patient_dataset.csv"
    X, y = preprocess(filepath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

    tree = DecisionTreeClassifier(random_state=1)
    tree.fit(X_train, y_train)
    tree_f1 = f1_score(y_test, tree.predict(X_test))

    bagging_clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=1),
        n_estimators=50,
        random_state=1
    )
    bagging_clf.fit(X_train, y_train)
    bagging_f1 = f1_score(y_test, bagging_clf.predict(X_test))

    print(f"Test set f1-score of aggregator: {bagging_f1:.2f}")
    print(f"Test set f1-score of single decision tree: {tree_f1:.2f}")
if __name__ == '__main__':
    main()