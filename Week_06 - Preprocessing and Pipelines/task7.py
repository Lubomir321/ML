import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main():
    df = pd.read_csv("music_clean.csv", index_col=0)

    X = df.drop(columns=["energy"])
    y = df["energy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

    linear_model = Pipeline([
        ('model', LinearRegression())
    ])

    ridge_model = Pipeline([
        ('model', Ridge(alpha=0.1, random_state=42))
    ])

    lasso_model = Pipeline([
        ('model', Lasso(alpha=0.1, random_state=42))
    ])

    kf = KFold(n_splits=6, shuffle=True, random_state=42)

    linear_scores = cross_val_score(linear_model, X_train, y_train, cv=kf)
    ridge_scores = cross_val_score(ridge_model, X_train, y_train, cv=kf)
    lasso_scores = cross_val_score(lasso_model, X_train, y_train, cv=kf)

    scores = [linear_scores, ridge_scores, lasso_scores]

    plt.boxplot(scores, labels=['Linear Regression', 'Ridge', 'Lasso'])
    plt.tight_layout()
    plt.show()

    #linear regression performs the best

if __name__ == '__main__':
    main()