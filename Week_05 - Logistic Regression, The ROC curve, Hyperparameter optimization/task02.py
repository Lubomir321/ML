import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def main():
    df = pd.read_csv("diabetes_clean.csv")

    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    clf = LogisticRegression(random_state=42).fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve for Diabetes Prediction')
    plt.show()

    # Answer: C
if __name__ == '__main__':
    main()