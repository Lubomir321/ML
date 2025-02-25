import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
    tree_entropy.fit(X_train, y_train)

    tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree_gini.fit(X_train, y_train)

    print(f"Accuracy achieved by using entropy: {tree_entropy.score(X_test,y_test)}")
    print(f"Accuracy achieved by using the gini index: {tree_gini.score(X_test,y_test)}")
if __name__ == '__main__':
    main()