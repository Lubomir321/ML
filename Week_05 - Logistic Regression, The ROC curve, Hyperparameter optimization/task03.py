import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay


def main():
    df = pd.read_csv("diabetes_clean.csv")

    X = df.drop(columns='diabetes')
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    clf = LogisticRegression(random_state=42).fit(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)


    print("Model KNN trained!")
    print("Model LogisticRegression trained!")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True) #why do they not share the same Y

    knn_probs = knn.predict_proba(X_test)[:, 1]
    knn_auc = roc_auc_score(y_test, knn_probs)
    print("KNN AUC:", knn_auc)
    print("KNN Metrics:\n", classification_report(y_test, knn.predict(X_test)))
    ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test, ax=axes[0])
    axes[0].set_title("KNN Confusion Matrix")


    log_reg_probs = clf.predict_proba(X_test)[:, 1]
    log_reg_auc = roc_auc_score(y_test, log_reg_probs)
    print("LogisticRegression AUC:", log_reg_auc)
    print("LogisticRegression Metrics:\n", classification_report(y_test, clf.predict(X_test)))
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=axes[1])
    axes[1].set_title("Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.show()
    #LogisticRegression, защото е предвидена да се използва масово за бинарна класификация

if __name__ == '__main__':
    main()