import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main():
    df = pd.read_csv("music_clean.csv", index_col=0)

    music_dummies = pd.get_dummies(df['genre'], drop_first=True, dtype=int)
    music_dummies = pd.concat([df, music_dummies], axis=1)
    music_dummies = music_dummies.drop(columns=['genre'])
    df = music_dummies
    df.rename(columns={df.columns[-1]: 'genre'}, inplace=True)

    X = df.drop(columns=['loudness'])
    y = df['loudness']


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    print("First 5 rows of X before scaling:")
    print(X_train.head())

    lasso1 = Lasso(alpha=0.5, random_state=42)
    lasso2 = Lasso(alpha=0.5, random_state=42)
    lasso_no_scaling = Pipeline([
        ('model', lasso1)
    ])

    lasso_with_scaling = Pipeline([
        ('scaler', StandardScaler()),
        ('model', lasso2)
    ])

    lasso_no_scaling.fit(X_train, y_train)
    lasso_with_scaling.fit(X_train, y_train)

    print(f"R^2 score without scaling: {lasso_no_scaling.score(X_test, y_test)}")
    print(f"R^2 score with scaling: {lasso_with_scaling.score(X_test, y_test)}")

    #Scaling helps the model perform better because lasso can focus more the on
    #the relationships between variables than on the magnitued of the varibales
if __name__ == '__main__':
    main()