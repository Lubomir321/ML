import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    df = pd.read_csv("diabetes_clean.csv")

    X = df.drop(columns=['diabetes'])

    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    clf = LogisticRegression(random_state=42).fit(X_train, y_train) #max_iter=1000 solved an issue i had 
    #D:\Anaconda\Lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    #STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    #or scaling the data
    print(clf.predict_proba(X_test[:10])[:,1])
if __name__ == '__main__':
    main()