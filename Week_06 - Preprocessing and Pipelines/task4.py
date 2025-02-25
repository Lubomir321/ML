import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    df = pd.read_json("music_dirty_missing_vals.txt")

    df['genre'] = df['genre'].apply(lambda x: 1 if x == "Rock" else 0)

    X = df.drop(columns=['genre'])
    y = df['genre']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('knn', KNeighborsClassifier(n_neighbors=3))   
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.tight_layout()
    plt.show()

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.legend(title=f'Classifier (AUC = {np.round(roc_auc,2)})', loc='lower right')
    plt.xlabel("False Positive Rate (Positive label: 1)")
    plt.ylabel("True Positive Rate (Positive label: 1)")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()