import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("advertising_and_sales_clean.csv")

    X = df.drop(columns=['sales', 'influencer'])
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    ridge_scores = {}

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        score = ridge.score(X_test, y_test) 
        ridge_scores[alpha] = score

    print("Ridge scores per alpha:", ridge_scores)


    plt.plot(ridge_scores.keys(), ridge_scores.values())
    plt.title("R^2 per alpha")
    plt.xlabel("Alpha")
    plt.ylabel("R^2")
    plt.yticks(np.linspace(0.99, 1, 10))
    plt.show()

    #"Do we have overfitting?"
    #Не, няма особена промяна
    #"Do we have underfitting?"
    #Не, няма особена промяна, дори и за големи алфи 
    #"How does heavy penalization affect model performance?"
    #Не променя значителни модела, може да се види от изменението на алфите

if __name__ == '__main__':
    main()