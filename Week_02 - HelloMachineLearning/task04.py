import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():
    df_churn = pd.read_csv('telecom_churn_clean.csv')

    X = df_churn.drop(columns=['Unnamed: 0', 'churn']).values

    y = df_churn['churn'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training Dataset Shape {X_train.shape}")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    print(f"Accuracy when 'n_neighbors=5': {round(knn.score(X_test, y_test), 4)}")

    train_accuracies = {}
    test_accuracies = {}

    neighbors = np.arange(1, 13)

    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n)
        
        knn.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, knn.predict(X_train))
        test_acc = accuracy_score(y_test, knn.predict(X_test))
        
        train_accuracies[n] = round(train_acc, 4)
        test_accuracies[n] = round(test_acc, 4)

    print(f"neighbors={neighbors}")
    print(f"train_accuracies={train_accuracies}")
    print(f"test_accuracies={test_accuracies}")

    plt.title('KNN: Varying Number of Neighbors')
    plt.plot(neighbors, train_accuracies.values(), label='Training Accuracy')
    plt.plot(neighbors, test_accuracies.values(), label='Testing Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    main()