import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    df = pd.read_csv("diabetes_clean.csv")

    X = df[['bmi', 'age']]
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    target_names = ['No diabetes', 'Diabetes']
    class_report = classification_report(y_test, y_pred, target_names=target_names)

    print("\nClassification Report:")
    print(class_report) 

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=target_names)

    plt.tight_layout()

    plt.show()
    #How many true positives were predicted? 33
    #How many false positives were predicted? 35
    #For which class is the f1-score higher?
    #Няма диабет precision = 0.711 Recall = 0.768 F1 = 0.738
    #Диабет precision = 0.485 Recall = 0.413 F1 = 0.446
    #F1 за Няма диабет
if __name__ == '__main__':
    main()