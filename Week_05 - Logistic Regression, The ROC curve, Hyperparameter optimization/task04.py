import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Lasso

def main():
    df = pd.read_csv("diabetes_clean.csv")

    X = df.drop(columns='glucose')
    y = df['glucose']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    lasso = Lasso(random_state=42)
    param_grid = {
        'alpha': np.arange(0.00001 , 1, 20),
    }
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    grid_search = GridSearchCV(lasso, param_grid, cv=kf)

    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']

    best_lasso = Lasso(alpha=best_alpha,random_state=42)
    best_lasso.fit(X_train, y_train)
    test_score = best_lasso.score(X_test, y_test)

    print("Tuned lasso parameters:", grid_search.best_params_)
    print("Tuned lasso score:", test_score)

    #Не е гарантирано. Стойността на R^2
    #зависи от сложността на данните и взаимовръзките между тях (линейна зависимост). 
    #Дори и с параметризация, може да не се достигне до добра оценка на R^2
if __name__ == '__main__':
    main()