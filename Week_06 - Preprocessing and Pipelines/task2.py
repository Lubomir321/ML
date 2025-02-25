import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.metrics import root_mean_squared_error

def main():
    df = pd.read_json("music_dirty.txt")
    df = pd.get_dummies(df, columns=['genre'], drop_first=True, dtype=int)

    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    model = Ridge(alpha=0.2)

    kf = KFold(n_splits=6, shuffle=True, random_state=42)

    cv_results = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error') 
    #https://scikit-learn.org/dev/modules/model_evaluation.html#scoring-parameter

    print(f"Average RMSE: {abs(sum(cv_results)/len(cv_results))}")
    print(f"Standard Deviation of the target array:  {np.std(y)}")
    #The model performs well because its RMSE is significantly smaller than the standard deviation
if __name__ == '__main__':
    main()