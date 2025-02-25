import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score

def main():
    columns = [
        "age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp", 
        "alb", "a/g_ratio", "has_liver_disease"
    ]
    df = pd.read_csv("indian_liver_patient_dataset.csv", header=None, names=columns)

    df.fillna(df.mode().iloc[0], inplace=True)

    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

    df["has_liver_disease"] = df["has_liver_disease"].map({2: 0, 1: 1})

    print("Data:\n")
    print(df)

    X = df.drop(columns=["has_liver_disease"])
    y = df["has_liver_disease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logreg = LogisticRegression(random_state=1)
    tree = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=1)
    knn = KNeighborsClassifier(n_neighbors=27)

    logreg.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    logreg_f1 = f1_score(y_test, logreg.predict(X_test))
    tree_f1 = f1_score(y_test, tree.predict(X_test))
    knn_f1 = f1_score(y_test, knn.predict(X_test))

    voting_clf = VotingClassifier(estimators=[
        ('logreg', logreg), ('tree', tree), ('knn', knn)
    ], voting='hard')
    voting_clf.fit(X_train, y_train)

    voting_f1 = f1_score(y_test, voting_clf.predict(X_test))

    print(f"\nLogistic Regression: {logreg_f1:.3f}")
    print(f"K Nearest Neighbours: {knn_f1:.3f}")
    print(f"Classification Tree: {tree_f1:.3f}")
    print(f"\nVoting Classifier: {voting_f1:.3f}")
if __name__ == '__main__':
    main()