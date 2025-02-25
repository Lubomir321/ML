from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

def main():
    digits = datasets.load_digits()

    print(f"Dataset shape: {digits.data.shape}")
    print(f"Number of classes: {len(np.unique(digits.target))}")

    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=42)

    random_indices = np.random.choice(len(X_train), 5, replace=False)
    random_samples = X_train[random_indices]
    random_labels = y_train[random_indices]

    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for ax, sample, label in zip(axes, random_samples, random_labels):
        ax.imshow(sample.reshape(8, 8), cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.show()

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)

    print(f"Training accuracy of logistic regression: {log_reg.score(X_train,y_train)}")
    print(f"Validation accuracy of logistic regression: {log_reg.score(X_test,y_test)}")

    svc = SVC(random_state=42)
    svc.fit(X_train, y_train)

    print(f"Training accuracy of non-linear support vector classifier: {svc.score(X_train, y_train)}")
    print(f"Validation accuracy of non-linear support vector classifier: {svc.score(X_test, y_test)}")

    #Базирайки отговора на теста за точност: svc

if __name__ == '__main__':
    main()