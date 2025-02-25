import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv("music_clean.csv", index_col=0)

    median_popularity = df['popularity'].median()
    df['popularity_binary'] = (df['popularity'] >= median_popularity).astype(int)

    X = df.drop(columns=['popularity', 'popularity_binary'])
    y = df['popularity_binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = [
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('KNN', KNeighborsClassifier()),
        ('Decision Tree', DecisionTreeClassifier(random_state=42))
    ]

    cv_results = {}
    kf = KFold(n_splits=6, shuffle=True, random_state=12) 

    for model_name, model in models:
        cv_score = cross_val_score(model, X_train_scaled, y_train, cv=kf)
        cv_results[model_name] = cv_score

    plt.boxplot(cv_results.values(), labels=cv_results.keys())
    plt.tight_layout()
    plt.show()
    #logistic regression performs best
if __name__ == '__main__':
    main()