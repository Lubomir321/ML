import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def main():
    df = pd.read_csv("music_clean.csv", index_col=0)
    print(df)
    X = df.drop(columns=['genre'])
    y = df['genre']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=21)

    logreg1 = LogisticRegression(random_state=21)
    logreg2 = LogisticRegression(random_state=21)

    param_grid = {'logreg__C': np.linspace(0.001, 1.0, 20)}

    pipeline_no_scaling = Pipeline([
        ('logreg', logreg1)
    ])

    pipeline_with_scaling = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', logreg2)
    ])

    grid_no_scaling = GridSearchCV(pipeline_no_scaling, param_grid, cv=5)
    grid_with_scaling = GridSearchCV(pipeline_with_scaling, param_grid, cv=5)
    #i was experimenting the the cv var to see which one gets the C hayperparameter from the the task
    grid_no_scaling.fit(X_train, y_train)
    grid_with_scaling.fit(X_train, y_train)


    print(f"Without scaling: {grid_no_scaling.score(X_test, y_test):.3f}")
    print(f"Without scaling: {grid_no_scaling.best_params_}")
    print(f"With scaling: {grid_with_scaling.score(X_test,y_test):.3f}")
    print(f"With scaling: {grid_with_scaling.best_params_}")

    #Without scaling, the accuracy is lower because features with larger magnitudes dominate the model, reducing its ability to make correct assumptions
if __name__ == '__main__':
    main()